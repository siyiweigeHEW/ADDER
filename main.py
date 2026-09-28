"""Frontend converter consistency audit - main program.

Workflow:
  0) Select the LLM
  1) Extract the frontend converters
  2) Code expansion (append the dependency code)
  3) First judgment: LLM consistency comparison
  4) Documentation retrieval
  5) Second judgment: LLM documentation-based deep audit

Run ``python main.py --backend openvino`` or ``--backend tvm``.
"""
import argparse
import os
from datetime import datetime

import Levenshtein

import settings
from code_expander import expand_code_body
from consistency_checker import compare_code_consistency
from doc_analyzer import analyze_with_docs, parse_final_verdict
from doc_retriever import get_doc_from_file
from extract_function import get_all_front_converters
from frontends import available, get_frontend
from llm_client import setup_model, global_model

SIMILARITY_THRESHOLD = 0.85


def record_detail(detail_file, info):
    with open(detail_file, "a", encoding="utf-8") as f:
        f.write(info + "\n" + "=" * 50 + "\n\n")


def record_pair_summary(summary_file, op_pair_name, code_match, doc_match):
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"{op_pair_name} {code_match} {doc_match}\n")


def find_lcsubstr(s1, s2):
    """Length of the longest common substring of ``s1`` and ``s2``."""
    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p]


def cal_text_sim(t1, t2, strategy="lcs"):
    t1 = t1.lower().replace("_", "")
    t2 = t2.lower().replace("_", "")
    if strategy == "ed":
        dis = Levenshtein.distance(t1, t2)
        text_sim = 1 - dis / max(len(t1), len(t2))
    elif strategy == "lcs":
        dis = len(find_lcsubstr(t1, t2))
        text_sim = dis / max(len(t1), len(t2))
    return text_sim


def get_similar_op_with_text_sim(op_name, target_dict):
    """Find the operator in ``target_dict`` most similar to ``op_name``."""
    op_name = op_name.split("::")[-1]
    max_sim = 0
    max_sim_op = None
    for target_op in target_dict.keys():
        _target_op = target_op.split("::")[-1]
        this_sim = cal_text_sim(op_name, _target_op)
        if this_sim > max_sim:
            max_sim = this_sim
            max_sim_op = target_op
    return max_sim_op, max_sim


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        default=os.environ.get("AUDIT_BACKEND"),
        help=f"tool stack under audit ({', '.join(available())}); "
             f"also settable via AUDIT_BACKEND",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fe = get_frontend(args.backend)
    tool_stack = fe.PROMPT_VALUES["ToolStack"]

    print("=" * 60)
    print(f"     {tool_stack} Frontend Consistency Audit System")
    print("=" * 60)

    # ========== Step 0: Select the LLM ==========
    print("\n[Step 0/5] Select the LLM model...")
    setup_model()

    # ========== Path configuration ==========
    front_op_dir_map = fe.front_dirs()
    front_names = list(front_op_dir_map)
    doc_map = settings.doc_paths(front_names)

    here = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(here, "results", global_model.model_name, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    detail_file = os.path.join(results_dir, "results_detail.txt")
    pairs_summary_file = os.path.join(results_dir, "pairs_result.txt")

    total_matched_pairs = 0
    skipped_count = 0
    equivalent_count = 0
    nonequivalent_count = 0
    skipped_consistency_count = 0
    doc_miss_count = 0          # doc not found (flag 8)
    inconclusive_count = 0      # doc analysis inconclusive/unparsed (flag 9)
    bug_candidate_count = 0     # doc analysis final verdict = Bug (flag 0)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting frontend consistency audit...")

    # ========== Step 1: Extract converters ==========
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [Step 1/5] Extracting frontend converters...")
    all_front_converter = get_all_front_converters(args.backend, fe.TARGET_FRONTS)
    all_front_name_list = list(all_front_converter.keys())

    for k, v in all_front_converter.items():
        print("\n" + "*" * 20 + f" Processing {k} frontend " + "*" * 20)
        api_map_dict = v[1]

        for op_name, func_name in api_map_dict.items():
            for target_front in all_front_name_list:
                if target_front == k:
                    continue

                similar_op_name, sim_score = get_similar_op_with_text_sim(
                    op_name, all_front_converter[target_front][1]
                )
                if sim_score < SIMILARITY_THRESHOLD:
                    continue

                total_matched_pairs += 1
                similar_api_name = all_front_converter[target_front][1][similar_op_name]
                op_pair_display_name = f"{k}.{op_name}_VS_{target_front}.{similar_op_name}"

                is_skip = (
                    "(" in similar_api_name
                    or "(" in func_name
                    or similar_api_name not in all_front_converter[target_front][0]
                    or func_name not in all_front_converter[k][0]
                )
                if is_skip:
                    skipped_count += 1
                    continue

                print(f"\nComparing: {k}.{op_name} <--> {target_front}.{similar_op_name} "
                      f"(similarity: {sim_score:.2f})")

                source_func_body = all_front_converter[k][0][func_name]
                target_func_body = all_front_converter[target_front][0][similar_api_name]

                # ========== Step 2: Code expansion ==========
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 2/5] Expanding code...")
                source_func_body, src_dep_info = expand_code_body(
                    source_func_body, k, front_op_dir_map[k], args.backend
                )
                target_func_body, tgt_dep_info = expand_code_body(
                    target_func_body, target_front, front_op_dir_map[target_front], args.backend
                )
                if src_dep_info or tgt_dep_info:
                    print(f"    [EXPAND] Code expanded: {k}=[{src_dep_info}], "
                          f"{target_front}=[{tgt_dep_info}]")

                try:
                    # ========== Step 3: First consistency judgment ==========
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 3/5] "
                          f"Running first consistency check...")
                    consistency_res, explanation, compare_consistency_prompt = \
                        compare_code_consistency(k, source_func_body, target_front,
                                                 target_func_body, args.backend)

                    if consistency_res == "nonequivalent":
                        nonequivalent_count += 1
                        code_match_flag = 0
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 3/5] "
                              f"Result: NOT equivalent, starting in-depth document analysis...")

                        # ========== Step 4: Document retrieval ==========
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 4/5] "
                              f"Retrieving documentation...")
                        doc_path_k = doc_map.get(k, doc_map[front_names[0]])
                        doc_path_target = doc_map.get(target_front, doc_map[front_names[0]])

                        doc_content_k = get_doc_from_file(doc_path_k, op_name)
                        doc_content_target = get_doc_from_file(doc_path_target, similar_op_name)

                        doc_missing = any(
                            c.startswith(("Warning:", "Error:"))
                            for c in (doc_content_k, doc_content_target)
                        )
                        op_pair_info = (f"{k}.{op_name} ({func_name}) <--> "
                                        f"{target_front}.{similar_op_name} ({similar_api_name})")

                        if doc_missing:
                            doc_match_flag = 8
                            doc_miss_count += 1
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 4/5] "
                                  f"Documentation not matched (Flag: 8).")
                            detail_log = (f"{'=' * 70}\n"
                                          f"Op Pair: {op_pair_info}\n"
                                          f"- code_match: {code_match_flag} | "
                                          f"doc_match: {doc_match_flag}\n"
                                          f"{'=' * 70}\n"
                                          f"\nFirst LLM (consistency) input:\n"
                                          f"{compare_consistency_prompt}\n"
                                          f"{'-' * 50}\n"
                                          f"\nFirst LLM output:\n{explanation}\n"
                                          f"{'-' * 50}\n"
                                          f"\nDoc retrieval:\n  {k}: {doc_content_k[:100]}\n"
                                          f"  {target_front}: {doc_content_target[:100]}")
                            record_detail(detail_file, detail_log)
                            record_pair_summary(pairs_summary_file, op_pair_display_name,
                                                code_match_flag, doc_match_flag)
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 5/5] "
                                  f"Skipping document analysis (Flag: {doc_match_flag}).\n")
                        else:
                            # ========== Step 5: Second document analysis judgment ==========
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 5/5] "
                                  f"Running second document analysis...")
                            final_judgment = analyze_with_docs(
                                explanation, doc_content_k, doc_content_target,
                                source_code=source_func_body, target_code=target_func_body,
                                backend=args.backend,
                            )

                            verdict = parse_final_verdict(final_judgment)
                            if verdict == "bug":
                                doc_match_flag = 0
                                bug_candidate_count += 1
                            elif verdict in ("standard_gap", "optimization"):
                                doc_match_flag = 1
                            else:  # inconclusive / unparsed
                                doc_match_flag = 9
                                inconclusive_count += 1

                            detail_log = (f"{'=' * 70}\n"
                                          f"Op Pair: {op_pair_info}\n"
                                          f"- code_match: {code_match_flag} | "
                                          f"doc_match: {doc_match_flag}\n"
                                          f"{'=' * 70}\n"
                                          f"\nFirst LLM (consistency) input:\n"
                                          f"{compare_consistency_prompt}\n"
                                          f"{'-' * 50}\n"
                                          f"\nFirst LLM output:\n{explanation}\n"
                                          f"{'-' * 50}\n"
                                          f"\nSecond LLM (doc analysis) output:\n{final_judgment}")
                            record_detail(detail_file, detail_log)
                            record_pair_summary(pairs_summary_file, op_pair_display_name,
                                                code_match_flag, doc_match_flag)
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 5/5] "
                                  f"Document analysis complete (Match Flag: {doc_match_flag}).\n")
                    elif consistency_res == "skipped":
                        skipped_consistency_count += 1
                        skipped_count += 1
                        record_pair_summary(pairs_summary_file, op_pair_display_name, 9, 9)
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 3/5] "
                              f"Result: Oversize, skipped (Flag: 9)\n")
                    else:
                        equivalent_count += 1
                        record_pair_summary(pairs_summary_file, op_pair_display_name, 1, 1)
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Step 3/5] "
                              f"Result: Logically equivalent\n")

                except Exception as e:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"[ERROR] LLM comparison failed: {e}")
                    skipped_count += 1

    print("\n" + "=" * 60)
    print("                Final Audit Report")
    print("=" * 60)
    print(f"  Backend: {tool_stack}")
    print(f"  Model: {global_model.model_name}")
    print("-" * 60)
    print(f"1. Total synonymous operator pairs (Sim >= {SIMILARITY_THRESHOLD}): "
          f"{total_matched_pairs}")
    print(f"2. Pairs compared successfully:                         "
          f"{total_matched_pairs - skipped_count}")
    print("-" * 60)
    print(f"   Logically equivalent (Equivalent):                   {equivalent_count}")
    print(f"   Judged non-equivalent (Nonequivalent):               {nonequivalent_count}")
    print(f"   Oversize skip:                                       {skipped_consistency_count}")
    print(f"   Documentation not matched (Flag 8):                  {doc_miss_count}")
    print(f"   Inconclusive doc analysis (Flag 9):                  {inconclusive_count}")
    print(f"   Other skips (pre-filter/exception):                  "
          f"{skipped_count - skipped_consistency_count}")
    print("-" * 60)
    print(f"   Bug candidates (doc_match=0):                        {bug_candidate_count}")
    print(f"   Results written to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# from eval import evaluate_prompt
# import asyncio
# if __name__ == "__main__":
#     results = asyncio.run(
#         evaluate_prompt(
#             "prompts/support_classifier_v1_balanced.yaml",
#             "src/data/golden_dataset.json",
#         )
#     )

#     print(results)
from eval import check_latest_accuracy_drop
check_latest_accuracy_drop(0.03)
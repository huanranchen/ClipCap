from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.cider.cider import Cider


def estimate_bleu(candidate, ground_truth):
    # candidate, ground_truth = test(total_num=total_samples)

    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(ground_truth, candidate)
    print(score)
    print('-' * 60)
    print(scores)

def estimate_cider(candidate, ground_truth):
    scorer = Cider(n=4)
    score, scores = scorer.compute_score(ground_truth, candidate)
    print(score)
    print('-' * 60)
    print(scores)


if __name__ == '__main__':
    estimate_bleu()

import joblib
import argparse
import os
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# The code is referenced from
# https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit


def get_args():
    parser = argparse.ArgumentParser(
        description="Learn K-means clustering over acoustic features.")
    parser.add_argument("--num-clusters", type=int,
                        help="Nubmer of clusters", default=512)  # K-Means默认512类，也就是论文中的V=512 token
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max-iter", type=int,
                        help="Maximum number of iterations for K-means training", default=150,)
    parser.add_argument("--batch-size", type=int,
                        help="Batch size for K-means training", default=512,)
    parser.add_argument("--tol", type=float, default=0.0,)
    parser.add_argument("--max-no-improvement", type=int, default=100,)
    parser.add_argument("--n-init", type=int, default=20, )
    parser.add_argument("--reassignment-ratio", type=float, default=0.5, )
    parser.add_argument("--channel-id", type=int, default=None, )
    parser.add_argument("--out-kmeans-model-path", type=str,
                        default=r'./km.bin', help="Path to save K-means model",)
    parser.add_argument('--kmeans-model-path', type=str, default=r'./km.bin', )
    # Leftovers
    parser.add_argument(
        "--seed", type=int, help="Random seed to use for K-means training", default=1046,)

    return parser.parse_args()


def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def train_kmeans(model, feaBatch):
    startT = time.perf_counter()
    model.fit(feaBatch)
    endT = time.perf_counter()
    print("Training time:", endT - startT, "s")

    return model


if __name__ == "__main__":
    args = get_args()

    # Prep training feats, and make batch
    # the place where to save wav2vec2 features
    readDir = "vctkDataset/wav2vec2_15th"
    feaBatch = []
    for file in os.listdir(readDir):
        readPath = os.path.join(readDir, file)
        feat = np.load(readPath)
        feaBatch.append(feat[0])
    feaBatch = np.vstack(feaBatch)  # [num_samples, num_features]
    print("The shape of features is:", feaBatch.shape)

    # Init model
    kmeans_model = get_kmeans_model(
        n_clusters=args.num_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
        random_state=args.seed)

    # Train
    kmeans_model = train_kmeans(kmeans_model, feaBatch)

    # save
    os.makedirs(os.path.dirname(args.out_kmeans_model_path), exist_ok=True)
    joblib.dump(kmeans_model, open(args.out_kmeans_model_path, "wb"))

from collections import defaultdict
import numpy as np


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap



def compute_topK(trn_hash, tst_hash, trn_label, tst_label, radius, number_class, topK=[100, 500, 1000]):
    smooth = 1e-7
    trn_label = np.argmax(trn_label, axis=-1)
    tst_label = np.argmax(tst_label, axis=-1)
    for k in topK:
        AP = defaultdict(list)
        AR = defaultdict(list)
        print("********** top {} ***********".format(k))
        for r in radius:
            for i in range(number_class):
                AP['AP{}'.format(r)].append([])
                AR['AR{}'.format(r)].append([])
        for i in range(tst_hash.shape[0]):
            query_label, query_hash = tst_label[i], tst_hash[i]
            distance = np.sum((query_hash != trn_hash), axis=1)

            argidx = np.argsort(distance)[:k]
            # precision=TP/(TP+FP) recall=TP/(TP+FN)
            buffer_yes = (query_label == trn_label[argidx])
            buffer_1_0 = np.stack([buffer_yes, 1 - buffer_yes])

            for r in radius:
                if r == 0:
                    TPFP = ((distance[argidx] == 0) * buffer_1_0).sum(axis=1)
                    FN = ((distance[argidx] != 0) * buffer_yes).sum()
                else:
                    TPFP = ((distance[argidx] <= r) * buffer_1_0).sum(axis=1)
                    FN = ((distance[argidx] > r) * buffer_yes).sum()

                AP['AP{}'.format(r)][query_label].append(TPFP[0] / (TPFP.sum() + smooth))
                AR['AR{}'.format(r)][query_label].append(TPFP[0] / (TPFP[0] + FN + smooth))
        for r in radius:
            ap, ar = [], []
            for j in range(number_class):
                ap.append(np.array(AP['AP{}'.format(r)][j]).mean())
                ar.append(np.array(AR['AR{}'.format(r)][j]).mean())
            print("AP{}: {}|{} AR{}: {}|{}".format(r, np.round(np.array(ap).mean(),4), np.round(np.array(ap).std(),4),
                                                   r, np.round(np.array(ar).mean(),4), np.round(np.array(ar).std(),4)))
        del AP, AR


def pearson_corr_batch(query_embedding, batch_embedding, eps=1e-7):

    query_embedding = query_embedding.astype(np.float32)
    batch_embedding = batch_embedding.astype(np.float32)

    X_centered = query_embedding - query_embedding.mean()

    Y_centered = batch_embedding - batch_embedding.mean(axis=1, keepdims=True)

    X_std = np.sqrt((X_centered**2).sum())
    Y_std = np.sqrt((Y_centered**2).sum(axis=1)) + eps

    numerator = Y_centered @ X_centered
    corr_list = numerator / (X_std * Y_std)

    return corr_list
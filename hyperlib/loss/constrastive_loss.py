import tensorflow as tf


def contrastive_loss(pos_embs, neg_embs, M, c=1.0, clip_value=0.9):
    '''
    The contrastive loss for embeddings used by Nickel & Kiela
    math::
        -\log( e^{-d(x,y)} / \sum_{n \in N(x)} e^{-d(x,n)})

    where (x,y) is a positive example,
    N(x) is the set of negative samples for x,
    and d(.,.) is the hyperbolic distance
    '''
    # clip embedding norms before expmap
    pos_embs = tf.clip_by_norm(pos_embs, clip_value, axes=2)
    neg_embs = tf.clip_by_norm(neg_embs, clip_value, axes=2)

    x_pos = M.expmap0(pos_embs, c)
    x_neg = M.expmap0(neg_embs, c)

    batch_loss = M.dist(x_pos[:,0,:], x_pos[:,1,:], c)

    x = x_pos[:,0,:]
    x = tf.expand_dims(x, 1)
    x = tf.broadcast_to(x, x_neg.shape)

    neg_loss = tf.reduce_sum(tf.exp(-M.dist(x, x_neg, c)), axis=1)
    # clip to avoid log(0)
    neg_loss = tf.clip_by_value(neg_loss, clip_value_min=1e-15, clip_value_max=1e10)
    batch_loss += tf.math.log(neg_loss)
    return tf.reduce_sum(batch_loss, axis=0)

# Source acknowledgement

**Intraday Volume Lab** is a maintained research portfolio adaptation by **Bikash Kushwaha**. Its starting point was a supplied academic notebook and related materials about intraday volume forecasting with DeepLOBv and CMEM.

The untouched source archive and academic submission documents are kept outside this public edition. This project presents the maintained implementation and its reproducible demonstration.

This edition consolidates the duplicated training code, corrects the cumulative-volume transformation and temporal validation, validates input data, adds numerical regression checks, and provides a synthetic example and reproducible report. Its trainable architecture retains the supplied notebook's convolutional encoder, context branch, stock embeddings, and dense regression layers. It does not implement the CMEM residual estimator described in the source reports.

No source license file accompanied the supplied archive. No project-wide license has been added that purports to cover all historical materials. Package dependencies retain their respective names and licenses.

## Research context

DeepLOB and CMEM are research references, not original inventions of this portfolio edition. The supplied materials reference:

- [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668).
- [Intra-daily Volume Modeling and Prediction for Algorithmic Trading](https://academic.oup.com/jfec/article-abstract/9/3/489/840918).

These references provide context. The included CNN is not claimed to be a complete reproduction of either research model. No live exchange affiliation, production trading capability, or real-data performance is claimed.

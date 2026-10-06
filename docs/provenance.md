# Development provenance

This independent implementation follows the equations and experiment design of
*Feature Information Dynamics in Diffusion*. The package has been newly organized
for this repository; public commands must work without the original research tree.

The root MNIST notebook is a teaching guide using a shared conditional U-Net
trained from scratch, with supplied selected weights and measured errors. It
explains the project page's class-information example; it is not the paper's
frequency-band experiment. Separate MLP and finite-prior references remain
explicitly labelled workflows.

Representation code is grouped by Pixel/JiT, RAE, SDVAE/SiT and
VAVAE/LightningDiT. SDVAE uses the May online posterior-sampling route; VAVAE
uses its original offline sampled-cache route. Native objectives, time sampling
and EMA are retained. [Third-party notices](../THIRD_PARTY_NOTICES.md) document
upstream source licenses, alongside the root MIT License.

Main hierarchy measurements use class-prompted masks and masked-Canny according to
the paper. Later ordinary-Canny diagnostics are a different condition definition.
The four representations use different official backbones; the paper's matched
LDiT-XL FID experiment and small-B recipe search are a separate experiment family.

Project-specific code uses the MIT License. Third-party data
and weights retain their own terms. Prepared conditions are available on [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics). The dataset page records availability of the optional VAVAE component. Code publication uses the reviewed current tree, with development archives excluded. Hashes establish file identity, not scientific correctness.

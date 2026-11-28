# G0 audit: M* BAGEL image edit (/v1/images/edits), 2026-11-25

Server: port 18302, GPU 2, configs/bagel_single_gpu_audit.yaml. Reference image identical in both requests (same red PNG, 512x512).

| label | window | ViT prefill_vit | vae_encoder prefill_vae | LLM prefill_vae | image_gen_guided3 steps | elapsed |
|---|---|---|---|---|---|---|
| i2i-ref-A-0 | 22:40:47 (contaminated) | 2 | 2 | 2 | 49 | 95.1 s |
| i2i-ref-A-1 | 22:42:26 (clean) | 1 | 1 | 1 | 49 | 97.6 s |

Contamination of i2i-ref-A-0: a duplicate edit request (server request #30, started 22:41:09) ran concurrently and was cancelled by client disconnect at 22:41:19. Its ViT and VAE encode calls fall inside the first window. The clean value is the second row.

Conclusion (from the clean row): M* re-runs vit_encoder and vae_encoder for the same reference image on a repeated edit. There is no encoder output reuse.

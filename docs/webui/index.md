# WebUI

Once started, access `http://localhost:8000` (or `<ip address>:<port number>` if you use remote hosting) to open the WebUI.

## How to use

### Building the TensorRT engine

1. Click on the "Engine" tab
   ![](../images/readme-usage-screenshot-01.png)
2. Enter Huggingface's Diffusers model ID in `Model ID` (ex: `CompVis/stable-diffusion-v1-4`)
3. Enter your Huggingface access token in `HuggingFace Access Token` (required for some repositories).
   Access tokens can be obtained or created from [this page](https://huggingface.co/settings/tokens).
4. Click the `Build` button to start building the engine.
   - There may be some warnings during the engine build, but you can safely ignore them unless the build fails.
   - The build can take tens of minutes. For reference it takes an average of 15 minutes on the RTX3060 12GB.

### Generating images

1. Select the model in the header dropdown.
2. Click on the `Generate` tab
3. Click `Generate` button.

![](../images/readme-usage-screenshot-02.png)

<div align="left">

# **Cortex.t Subnet** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---

---
- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)

## Introduction

**IMPORTANT**: If you are new to Bittensor, please checkout the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section. 

The Bittensor Subnet 18 (cortex.t) is designed to provide reliable consistent quality responses for  app developemnt via api usage through the bittensor protocol [BitAPAI](https://bitapai.io). It is also designed to provide an accessible, fair and manipulation free landscape to the incentivised production of data (mining) and reward production of organic user prompts. 

This initiative takes the first steps in simplifying and re-imagining how text-prompting can be rewarded and is a push to provide stability and reassurance to developers of api-related apps and products allowing them to prioritise the provision of value to their clients without the worry of data inconsistencies. 

"Why is it not as good as ChatGPT?" was the often made comparison. Now it is. Now you can rely on the quality of GPT within the Bittensor network and pair it with other subnets and modalities using a single API key from [BitAPAI](https://bitapai.io).

## Setup

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- **IMPORTANT**: We **strongly recommend** before preceding that you test both subtensor and openai api key. Ensure you are running subtensor locally to minimize chances of outages and improve the latency/connection. After exporting your Openai api key to your bash profile, test the streaming service for both the gpt-3.5-turbo and gpt-4 engines using ./neurons/test_openai.py. Neither the miner or the validator will function without a valid and working openai api key. 

- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for cortex.t. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that this subnet requires very little compute. We recommend to run the miner and/or validator on a machine that you are already using for a different subnet. The main functionality is just api calls, so we outsource the compute to openai. The cost for this subnet comes from api calls, not from compute.

### Installation
Once you are navigated into the repo, `pip install -e .` will install the necessary requirements
In order to run a miner or validator, make sure you set your openai key to your profile.
```echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command `btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]`.

### Mining
In order to start miners easily with pm2, we have made a bash [alias](./alias.sh) for 10 miners and a validator. Note that for mining this script assumes your hotkeys are named 01-09. If you don't follow this naming convention, make sure to update the `hotkey` variable in the `start_miner` function. Ensure all other preferences align with the other variables within that file. After that, just run `source alias.sh` and then you can run the commands `start_miner[n]` with n being hotkey number or `start_vali` to launch a pm2 process that should score perfectly on the network.

### Validating

Similarly to the miner, just fix the variable names in the [alias](./alias.sh) script, source it, and then launch by running `start_vali` in your terminal. Simple as that!

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

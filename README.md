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

The Bittensor Subnet 18 (cortex.t) is designed to provide reliable consistent quality text responses for app developemnt via API usage through the bittensor protocol. It is also designed to provide an accessible, fair and manipulation free landscape to the incentivised production of data (mining) and reward production of organic user prompts. 

This initiative takes the first steps in simplifying and re-imagining how text-prompting can be rewarded and is a push to provide stability and reassurance to developers of API-related apps and products allowing them to prioritise the provision of value to their clients without the worry of data inconsistencies. 

"Why is it not as good as ChatGPT?" was the often made comparison. Now it is. Now you can rely on the quality of GPT within the Bittensor network and pair it with other subnets and modalities using a single API key from any validator, [BitAPAI](https://bitapai.io), or by building directly on an existing validator.


## Setup

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

**IMPORTANT**: We **strongly recommend** before proceeding that you test both subtensor and OpenAI API keys. Ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection. If you are unable to run Subtensor locally then you can also use the Taostats Subtensor endpoint by appending the followin to your start commands for a miner or validator.

```--subtensor.chain_endpoint wss://bittensor-finney.api.onfinality.io/public-ws --subtensor.network local```

After exporting your OpenAI API key to your bash profile, test the streaming service for both the gpt-3.5-turbo and gpt-4 engines using ```./neurons/test_openai.py```. Neither the miner or the validator will function without a valid and working [OpenAI API key](https://platform.openai.com/). 

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for cortex.t. See the [Minimum compute YAML configuration](./min_compute.yml).
Note that this subnet requires very little compute. The main functionality is api calls, so we outsource the compute to openai. The cost for mining and validating on this subnet comes from api calls, not from compute. Please be aware of your API costs and monitor accordingly.


### Installation

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```git clone https://github.com/BitAPAI/cortex.t.git && cd cortex.t && pip install -e .```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command `btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]`.

In order to run a miner or validator you must first set your OpenAI key to your profile with the following command.

```echo "export OPENAI_API_KEY=your_api_key_here">>~/.bashrc && source ~/.bashrc```


## Mining

You can launch your miners via pm2 using the following command. 

`pm2 start ./neurons/miner.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>`


## Validating

You can launch your validator via pm2 using the following command.

`pm2 start ./neurons/validator.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>`


## Logging

As cortex.t supports streaming natively, you do not (and should not) enable `logging.trace` or `logging.debug` as all of the important information is already output to `logging.info` which is set as default.

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

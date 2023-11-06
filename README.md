<div align="left">

# **Cortex.t Subnet** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
---

---
- [Introduction](#introduction)
- [Setup](#setup)
  - [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Introduction

**IMPORTANT**: If you are unaware of Bittensor, please checkout the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section. 

Explain why this subnet is important, brief overview of what it does. 

## Setup

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- **IMPORTANT**: We **strongly recommend** before preceding that you test both subtensor and openai api key. Ensure you are running subtensor locally to minimize chances of outages and improve the latency/connection. After exporting your Openai api key to your bash profile, test the streaming service for both the gpt-3.5-turbo and gpt-4 engines using ./neurons/test_openai.py. Neither the miner or the validator will function without a valid and working openai api key. 

- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for cortex.t. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that this subnet requires very little compute. We recommend to run the miner and/or validator on a machine that you are already using for a different subnet. The main functionality is just api calls, so we outsource the compute to openai. The cost for this subnet comes from api calls, not from compute.

### Installation


### Starting a Miner

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/running_on_staging.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).


### Starting a Validator

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/running_on_staging.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).

---

### Examples

The Bittensor Subnet 1 for Text Prompting is built using this template. See [Bittensor Text-Prompting](https://github.com/opentensor/text-prompting) for how to configure the files and how to add monitoring and telemetry and support multiple miner types. Also see this Subnet 1 in action on [Taostats](https://taostats.io/subnets/netuid-1/) explorer.

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

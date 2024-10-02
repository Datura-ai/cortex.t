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

## Setup

### Before you proceed
Before you proceed with the installation of the subnet, note the following:

We **strongly recommend** before proceeding that you test both subtensor and all API keys. Ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection.

### API Key Requirements

API requirements for this subnet are constantly evolving as we seek to meet the demands of the users and stay up to date with the latest developements. The current key requirements are as follows:

- OpenAI key (GPT)
- Google API key (Gemini)
- Anthropic API key (Claude3)
- Groq API key (Llama, Gemini, Mistral)
- AWS Acces key and Secret key (Bedrock models)
- Pixabay API key

Please test the API keys from ```.env.example``` with ```test_scripts/t2t/``` before starting the miner/validator. Detailed instructions on how to aquire the API keys below.

### Requesting Access for AWS Bedrock Models

#### 1. AWS Account
Ensure you have an active AWS account. If you don't have one, you can create it at [AWS Account Creation](https://aws.amazon.com/).

#### 2. Sign In to AWS Management Console
Go to the [AWS Management Console](https://aws.amazon.com/console/) and sign in with your AWS credentials.

#### 3. Navigate to AWS Bedrock
- In the AWS Management Console, use the search bar at the top to search for "Bedrock".
- Select AWS Bedrock from the search results.

#### 4. Request Access
- If AWS Bedrock is not directly available, you might see a page to request access.
- Follow the prompts to fill out any required information. This might include your use case for the models, your AWS account ID, and other relevant details.

#### 5. Submit a Request
- Complete any forms or questionnaires provided to describe your intended use of AWS Bedrock models.
- Submit the request for review.

#### 6. Wait for Approval
- AWS will review your request. This can take some time depending on the specifics of your request and the current availability of AWS Bedrock.
- You will receive an email notification once your request is approved or if further information is needed.

### Obtaining AWS Access Key and Secret Key

#### 1. Sign In to AWS Management Console
Go to the [AWS Management Console](https://aws.amazon.com/console/) and sign in with your AWS credentials.

#### 2. Navigate to My Security Credentials
- Click on your account name at the top right corner of the AWS Management Console.
- Select "Security Credentials" from the dropdown menu.

#### 3. Create New Access Key
- In the "My Security Credentials" page, go to the "Access keys" section.
- Click on "Create Access Key".
- A pop-up will appear showing your new Access Key ID and Secret Access Key.

#### 4. Download Credentials
- Download the `.csv` file containing these credentials or copy them to a secure location.
  - **Important**: This is the only time you will be able to view the secret access key. If you lose it, you will need to create new credentials.

### 5. Alternative - create dedicated user (more secure)
- Navigate to IAM
- Create New User - name the user `sn-18` or similar
- Attach `AmazonBedrockFullAccess` policy to user or apply the following permissions
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockAll",
            "Effect": "Allow",
            "Action": [
                "bedrock:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "DescribeKey",
            "Effect": "Allow",
            "Action": [
                "kms:DescribeKey"
            ],
            "Resource": "arn:*:kms:*:::*"
        },
        {
            "Sid": "APIsWithAllResourceAccess",
            "Effect": "Allow",
            "Action": [
                "iam:ListRoles",
                "ec2:DescribeVpcs",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups"
            ],
            "Resource": "*"
        },
        {
            "Sid": "PassRoleToBedrock",
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/*AmazonBedrock*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": [
                        "bedrock.amazonaws.com"
                    ]
                }
            }
        }
    ]
}
```

### Obtaining API Key from OpenAI

#### 1. OpenAI Account
Ensure you have an active OpenAI account. If you don't have one, you can create it at [OpenAI Account Creation](https://platform.openai.com/signup).

#### 2. Sign In to OpenAI
Go to the [OpenAI Platform](https://platform.openai.com/api-keys) and sign in with your OpenAI credentials.

#### 3. Create New API Key
- Click on the "Create new secret key" button.
- Follow the instructions provided to create your API key.


### Obtaining API Key from Google AI Platform

#### 1. Sign In to Google AI Platform
Go to the [Google AI Platform](https://aistudio.google.com/) and sign in with your Google credentials.

#### 2. Get API Key
- In the Google AI Platform, click on the "Get API key" button at the top left corner.
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Anthropic

#### 1. Anthropic Account
Ensure you have an active Anthropic account. If you don't have one, you can create it at [Anthropic Account Creation](https://www.anthropic.com/signup).

#### 2. Sign In to Anthropic
Go to the [Anthropic Platform](https://console.anthropic.com/settings/keys) and sign in with your Anthropic credentials.

#### 3. Get API Key
- In the Settings, go to the "API keys" tab and click on the "Create key" button at the top right corner.
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Groq

#### 1. Groq Account
Ensure you have an active Groq account. If you don't have one, you can create it at [Groq Account Creation](https://groq.com/signup).

#### 2. Sign In to Groq
Go to the [Groq Platform](https://console.groq.com/) and sign in with your Groq credentials.

#### 3. Get API Key
- In the Groq Platform, click on the "API keys" button at the left side.
- Click "Create API key"
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Pixabay

#### 1. Pixabay Account
Ensure you have an active Pixabay account. If you don't have one, you can create it at [Pixabay Account Creation](https://pixabay.com/ru/accounts/register/).

#### 2. Sign In to Pixabay
Go to the [Pixabay API docs](https://pixabay.com/api/docs/) and sign in with your Pixabay credentials.

#### 3. Get API Key
- Scroll down this page a little. Your key will be highlighted in green in the parameters for one of the requests.


### Installation

Before starting make sure update your system and have pm2 installed to run the scripts in the background.

```bash
apt update -y && apt-get install git -y && apt install python3-pip -y

```

Download the repository, navigate to the folder and then create virtual env and install the necessary requirements with the following chained command.

```bash
git clone https://github.com/corcel-api/cortex.t.git && cd cortex.t && pip3 install -e .
```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command
```bash
btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]
```

After installing it, copy `env.example` to `.env` and substitute
all env vars with values appropriate for your accounts.

## Mining
# step1.
go to cortext/constants.py and change bandwidth_to_model value as per limit of api.
currently we support only 3 models: "gpt-4o", "claude-3-5-sonnet-20240620", "llama-3.1-70b-versatile". 
so don't add more than that.
You can launch your miners via python3 using the following command.
```bash
bash start_miner.sh
```


## Validating

Login to wandb using

```bash
wand login
```

You can launch your validator using following command

```python
pm2 start start_validator.py --interpreter python3 -- --wallet_name "default" --wallet_hotkey "default" --subtensor.chain_endpoint <URL here> --autoupdate --wandb_on
```
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

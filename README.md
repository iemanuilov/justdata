# JUST Data Annotation Tool

![JUST Logo](images/logo.png "JUST Logo")

This repository contains a Python implementation of the JUST Data Annotation tool which allows the user to load and annotate datasets using a set of predefined tags as well as user-generated tags and attach use case specific files.

JUST Data Annotation focuses on developing a digital workflow fostering gender equality and inclusivity in data collection, handling and management, particularly within the context of industrial data and AI applications. JUST – Judicious, Unbiased, Safe and Transparent – is not intended as a metric, but aims to set guidelines for data literacy about bias, sensitive data, the context of data collection and data provenance. Our methodology includes:
* Creating standards for data annotation that prioritise fairness, safety, and transparency.
* Engaging a wide range of participants, including underrepresented groups, in the development and testing phases to ensure diverse perspectives are considered.
* Prototyping new standards and formulating practical guidelines for businesses to implement JUST data principles in their operations.
* Establishing testbed environments to trial, iterate and refine the JUST data annotation processes, ensuring they are effective and applicable across various industry sectors.

The fronted was built using ```streamlit``` to create a scriptable web app. The backend was built using ```PostgreSQL```, ```JSON```, ```pandas```, ```langchain```, ```openai``` and other popular Python libraries.

## Installation
Clone the repository:

```git clone https://github.com/IndustryCommons/justdata.git```

Install the necessary requirements, run:  

```pip install -r requirements.txt```

## Usage
To run on a local server, execute:  

```streamlit run justdata.py --server.enableCORS false --server.enableXsrfProtection false```

The app has been deployed on the Streamlit Community Cloud [here](https://justdatatool.streamlit.app/).

## Get in touch
If you need more information or would like to contribute to JUST Data Annotation, please contact us at info@industrycommons.net

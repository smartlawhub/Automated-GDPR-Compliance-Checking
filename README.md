# Automated-GDPR-Compliance-Checking

This repository contains code and data used to train the models mentioned in the paper "A combined rule-based and machine learning approach for automated GDPR compliance checking"  presented at ICAIL 2021.

We developed methods to automate compliance checking of privacy policies. We test a two-modules system, where the first module relies on NLP to extract data practices from privacy policies. The second module encodes GDPR rules to check the presence of mandatory information.

We make use of the OPP-115 dataset for training and evaluation of our models. We treat the extraction of data practices as a Hierarchical Multi-label Classification (HMTC) task and experiment with two different approaches: local classifiers and text-to-text. Our proposed text-to-text method has several advantages over local classifiers, including extraction of additional information and better scalability.

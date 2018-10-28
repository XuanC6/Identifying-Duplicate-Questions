# DupQues
INTRO TO MACHINE LEARNING PROJECTS: IDENTIFYING DUPLICATE QUESTIONS

## Background and summary: 
This dataset was published by Quora for the purpose of solving the problem of identifying duplicate questions to simplify searching for answers to a question posed. As a simple example, the queries “What is the most populous state in the USA?” and “Which state in the United States has the most people?” should not exist separately on Quora because the intent behind both is identical. Having a canonical page for each logically distinct query makes knowledge-sharing more efficient, so that knowledge seekers can access all the answers to a question in a single location.

## Goal: 
Given a sentence pair, identify if the sentences are semantically equivalent - that is, if the sentences are duplicates.

## Input data: 
Over 400,00 lines of sentence pairs:

1. qid1, quid2: ID of question 1, 2

2. question1, question2: Text of each question

3. is_duplicate: Binary true/fase label indicating if the line is a duplicate pair

Data can be found here: https://www.kaggle.com/quora/question-pairs-dataset

## Relevant Research:

https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

Paraphrase Detection

Semantic Similarity

Textual Entailment

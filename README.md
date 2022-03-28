# Fine tune GPT2 with DeepSpeed to generate LinkedIn-style job postings using just title and keywords

Generated with GPT Neo 1.3B


>Data Scientist - Recommendation systems
>Review our client!
>
>This is the perfect opportunity for an extraordinary individual to make an impact. Your technical expertise will be put into the hands of millions of users on a daily basis.
>
>The Requirements:  
>
>Experience building practical recommendation and machine learning systems
>Strong problem solving skills with experience crafting solutions, performing manual data reviews, and using statistics to improve models.
>Practical exposure to state of the art deep neural networks (Neural Networks) and related technologies
>Well versed in modern recommender engines and their architecture 
>You have designed software applications involving on-device processing / prediction
>Preferred but not required TensorFlow or PyTorch
>Your problem space may be complex
>You enjoy getting your solutions out there on NNs. You get feedback from expert colleagues who are experts.
>
>Day-to-Day Relevance
>
>The Data Science team are looking for a recommendations-oriented Senior Software Engineer in Datacenters, Analytics & Reporting for Recommendations as well as ML / Python Developers. In this role you’ll drive real time model generation/inference based reporting, dashboards and analysis pertaining to Recommender Systems such as Nearest Neighbor Search. This is the perfect opportunity to make those recommendations happen via dataset based analytics – ideal candidates love precision medicine aspect of problems they study (ie. Anandabound).
>
>Responsibilities:
>Be ready to deploy simple machine learning systems.
>Analyze large volumes of text at scale; work fast and accurate by drawing on existing datasets and using advanced modeling algorithms like Long Short Term Memory for classification and Regression.  
>Collaborate with data scientists and engineers from datacomputing teams to design pipeline of analytical models and develop supporting tools like code management processes optimizing code runtimes etc.


## Intro
It's known that GPT2-like models (GPT2 or GPT Neo) are capable of generation of texts from a string prompt. This method has no means of controlling content of the generated text. Addition of key- and key-phrases extracted from the text allows better control of the generated text.

This small project contains set of scripts to fine tune GPT2 models (L and XL) and generate the exemplary LinkedIn-styled job ads. 

## Fine Tuning procedure
This section describes training data preparation, tuning method and usage of the tuning script.

### Training data description
The fine-tuning is done on job postings on LinkedIn that were scrapped during 2021 with my own scrapper. Approximately 5000 unique job postings were scrapped. The texts were cleaned from artifacts before they were processed to extract 1 - 3 n-grams. The n-grams were extracted using MMR (maximal marginal relevance) of the n-gram and sentence embeddings. The processed data was then split into train-validation sets under condition that the texts contain no more than N tokens and no less than K. The results are locate in **input_data** folder. The files for convenience are pickled Pandas DataFrames.

### Fine tuning
#### Customized tokenizer
As the first step of the fine-tuning a tokenizer is set with addition of separator token. This token will separate the title from the keywords and the training text. For example, a job title *Data Scientist - Recommendation systems* and keywords *Recommendation systems,Python,neural networks,develop algorithm,predict customer interest,TensorFlow,ML frameworks* will be separated by the <|SEP|> token. 

The tokenizer is set to pad any input sequence that is shorter than **max_len** parameter (in a training config) and truncate anything that is longer than this. Due to limits of available VRAM, the texts should be limited in length. Many LinkedIn job postings have lengths well over 1000 tokens. In my experience, texts with 800+ tokens encompass roughly 1/4 of the total dataset of almost 5000 job postings. Even with DeepSpeed and moderate 12GB VRAM GPU this results in OOM errors when a larger models like GPT2 L or XL are tuned. Therefore a simply truncated dataset will contain a substantial fraction of incomplete texts. This will bias the model towards generation of incomplete and broken postings. A better approach is to filter job postings that are less than *N* tokens large. 

 **max_len** should be chosen larger than the maximal number of tokens in the training data to accommodate most of text. Since majority of the text contain fairly common (sometimes even looking similarly) endings on employer being equal opportunity, non-discriminative, etc, the generalization of the tuned model should not be seriously affected. In the same time it allows to feed larger texts into the model that often contain interesting texts.

#### Training
The fin tuning is implemented in ```cli_training.py -cfg config_name``` script. The fine tuning parameters are written in a .json file. The real training configs and DeepSpped (ds) configs are locate in **generate_configs** and **DS_configs** respectively. 
Refer to **NLP_model class** of the **utils.py** file for technical implementations. 

For training in a Jupyter Notebook on a local machine, use these parameters:
```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1" 
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

#### Text generation
Once a model is fine-tuned, it can generate text from a prompt. The prompt is formed from a job title (e.g. *Data Scientist - Recommendation systems*) and keywords separated by comma. The actual generation is implemented in ```cli_generate_text.py -cfg config_name```. The sample configs are located in **generate_configs/**. Some sample texts can be found in **generated_texts/**

### Results
Let’s compare results from GPT Neo (1.3B) and GPT2-Large for prompts from topics that were not present in the training datasets. The texts below were generated from these inputs:
- *Title*: Natural Language Processing (NLP)
- *Key words*: NLP,transformers,BERT,GPT,Python,customer satisfaction,analyze customer feedback,topic modeling,working with real-world text corpora


**GPT Neo**

>Description
>We are working with real world text that will enable smart search. Our goal is to make Amazon search useful for all its customers not only Amazon’s own ESLP initiatives focused on Smart Order Experience but also enabling future orders on any other major e-Commerce platforms by including Natural Language Understanding (NLP). At the same time we want them to be an amazing partner helping us build and run high performance ML systems and service for AWS in order to be ready for our next step of transform them from a research prototype to production. This requires that we continue inventing on behalf team members as well as building up new systems along the way with cutting-edge technology, state of art artificial intelligence, NLP methodologies at top tech companies. We create scalable and dynamic NLP solutions, and implement end-to-end ML solutions into these systems as needed, ensuring the appropriate level of accuracy when deploying trained models over live production systems.
>
>Our researchers and engineers love coming together across teams. They come from top universities such as Stanford University in EHS and EEFT, CUpertino and NYU Langone in HEP, Statute labs in NJM, NIPS Labs in MDL, USC IAF and TechCrunch in Singapore, etc.
>
>The ideal candidate's favorite words are learning, innovation, scale and agility. You will leverage your existing technical expertise and bring prototypes from previous open-source efforts into our systems through contributions. In addition, you will leverage previous NLP related success experiences within successful BERT like JelinekB, TalkPort, DNNube or others, in order to transfer these techniques into current SOH and natural language understanders space as business priorities change and evolve.
>
>
>Basic Qualifications
>Experience programming in Java, C++, Python or similar language
>2+ years professional software development experience where you have worked on some kind of project end
>Strong problem solving ability, and an ability to effectively articulate and communicate this complex information to non-technical audiences
>PhD/SAS in Computer science (or a related discipline, if you had an example in your schoolwork) is strongly preferred
>Significant experience creating and using custom modeling techniques on large datasets and applied problems
>Demonstrated leadership abilities in developing and leading technical discussions with senior colleagues throughout various dimensions.is expected to contribute high quality results to all aspects of modeling – not just for their assigned projects, though
>
>
>Preferred Qualifications
>Background in computational linguistics
>Background in topic modeling
>Knowledge in core areas of statistics: hypothesis testing, sampling & collecting, analyzing distributions, multi-variance analysis, etc. is definitely a plus
>Expertise with NLP technologies: tokenization, NLU tasks, classification, sentiment analysis tasks, question answering, textual similarity, transformer models, etc.
>A strong track record of thoughtful leadership in small, collaborative teams


>Company - Amazon.com Services LLC
>Job ID: A115789


**GPT2 large**

>Natural Language Processing (NLP)
>The Opportunity
> 
>We are looking for a motivated Principal Data/Speech Engineer to join us in Atlanta. In this role, you’ll have the unique and rewarding opportunity of working on natural language technology that impacts billions of customers daily.
>
>This is an outstanding hire for someone who is passionate about machine learning, NLP and wants to innovate the way we research & deploy large scale services using ML algorithms. It also includes some cross discipline design work as well.
>  You'll be working from our office across the street while having remote hours Monday thru Saturday. You will support a very fast growing product and deliverables team utilizing cutting edge Natural Language processing technologies using latest GPT architecture. At times in the evenings we might even sit together outside!
>
>What We're Looking For
>You hold 4+ years in computer science or related field
>Experience delivering services leveraging Natural Language technology
>Excellent data ingestion experience
>Knowledge of Deep Learning concepts including topic modeling, model inference, transformer models etc., such as: transformer learning, word embeddings or sentiment analysis
>Ability to multi task, communicate clearly, extract meaning out of ambiguous situations, work independently with little supervision
>BS / MS preferred
>
>Our Customers
>People in retail, automotive, hospitality and financial services use our products every day
>Who They Serve
>CVS.com, CVS Health, Aetna, American Express, Bayshore Financial Services, BB&T Mutual Funds, Barclays Capital Markets, Centura Federal Savings Bank, CenterCity Financial Group, Credit Unions International Inc, Delta Airlines Corporation, eBay LLC, Expedia Americas - Our largest online destination for travel, shopping, dining experiences, lodging and more
>Customer Satisfaction - 93% Customer Satisfaction!
>
>Employment-Type Summary
>Participates in R&D projects in close collaboration with management and other engineering teams throughout GE Digital Services unit
>Analyzes problem cases to provide solution recommendations in non technical domains and translate those solutions into technology concepts to solve real world problems
>Develops software platforms on top of the GE GPT platform, typically working with production services
>Manages expectations in terms of timelines, responsibilities and costs in meetings
>
>Qualifications
>Currently working on Natural Language Understanding, document indexing or summarization based algorithms or other applications requiring high level NLP systems development
>Bachelors and/or PhD in Computer Science with focus in the application domain


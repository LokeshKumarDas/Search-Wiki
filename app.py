# In this project i created a streamlit app to generate a text summary of given query(Using wikipedia as source)
# webscrapping for data collection and to summarize the data i used T5(small) Transformer(Text-to-Text Transfer Transformer) 

# importing libraries
from googlesearch import search    # for searching urls
import requests                    # to request the server data 
import re                          # to match urls with wikipedia urls
from bs4 import BeautifulSoup      # to extract data from HTML doc
import numpy as np                 # create a new 1D object(array)
from lxml import html              # to parse HTML
from transformers import T5Tokenizer, T5ForConditionalGeneration  # for text generation and summarization
import streamlit as st             # to represent out output

# creating filter to get only wikipedia sites
# as we want relible sources for collecting information
def filter_wikipedia_sites(url_list):
    wikipedia_sites = []
    for url in url_list:
        # for every url if it belongs to wikipedia it will be stored for further operations
        if re.match(r'https://en.wikipedia.org/wiki/.*', url):
            wikipedia_sites.append(url)
    return wikipedia_sites

# we will get all the content from wikipedia
def scrape_wikipedia_page(url):

    # taking the response from url (200: successful, 404: error)
    response = requests.get(url) 

    if response.status_code == 200:
        # taking all the HTML code from url
        soup = BeautifulSoup(response.text, 'html.parser')
        # every important information in wikipedia is a part of 'mw-parser-output' class and 'p' tag
        # extract and return text from paragraphs
        content = soup.find('div', {'class': 'mw-parser-output'})
        paragraphs = content.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])

        return text

    else:
        # if error occured
        return "Page not found."

# generating the summarized text
def summarize_with_t5(text, max_length=150, min_length=80):
    # prepend "summarize:" as T5 expects task-related prefixes
    input_text = "summarize: " + text
    
    # Tokenize the input text
    # max_length is takes 512 to represent every word with best description while can be adjusted
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # generate summary with the model
    # adjusting max and min length for a longer output
    # num_beams helps deciding the quality of summarization (we need fast but effecient)
    # length_penalty to support longer sequences
    
    # Source for adjusting parameters: https://huggingface.co/docs/transformers/en/main_classes/text_generation
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=5.0, num_beams=20, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary



# designing streamlit app

# title of our web-application
st.title('Search Wiki . . . ')

# getting search query for finding relevant URLs
query = st.text_area("Enter some keywords and search wikipedia ", height=1)

# Fetch URLs from search results
search_results = search(query)

# Number of search results to consider
# 5 is considered optimum while depending upon requirements it can be increased or decreased
num_results = 5

# collecting the sites containing relevent information
sites = np.fromiter(search_results, dtype = 'U100', count = num_results)

# filter only wikipedia sites
sites = filter_wikipedia_sites(sites)

# collecting all the data from sites collected
content = " "
for url in sites:
  content = content + " " + scrape_wikipedia_page(url)

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Use 't5-small' for faster results
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

# piplining the output summarized text
w = 0 

# logic :
# if Search button is clicked, we need to show the output
# first we will present the source of information (sites used in web scrapping)
# now we want to display the summarized text
# because we want to generate fast response we break the content into 200 batch size and generated summary of every batch
# hence serially generating the summary will give more faster and better results 

if st.button("Search"): 
  # will be activated when 'Search' button is clicked
  st.subheader('Source :-')
  if sites:
    # if sites are fetched successfully display all sites
    for site in sites:
      st.write(f'-> {site}')
  else:
    # if sites are not fetched successfully
    st.write('Not enable to fetch Wikipedia, Please try some other keywords.')

  st.subheader('Some Facts :-')
  
  if (int(len(content)/100)+1) <= 1:
        # because we didn't get much info from web-scrapping, changing some keywords will help
        # if words are less than 100 we will show not much information 
    st.write(f' Looks like, not able to fetch much info from wikipedia.')
    st.write(f' Please try with some other keywords !')
  else:
        # serialwise generating summary of every 200 words from content
        # taking max_length=70 and min_length=40 was giving most decent results
    for i in range(int(len(content)/200)):
      # generating summary of 200 words
      text_summary =summarize_with_t5(content[w:w +200], max_length=70, min_length=40)  
      w = w + 200
      # display text summary 
      st.write(f'{text_summary}')


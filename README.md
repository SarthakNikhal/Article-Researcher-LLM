## Article Research Tool

This tool makes it easy to research news. Just enter an article link and ask questions to get useful insights about the stock market and finance.

#### Use
1. Enter URLs or upload text files with URLs to get article content.
2. Use LangChain’s UnstructuredURL Loader to process the articles.
3. Create an embedding vector with OpenAI’s embeddings and use FAISS, a tool that helps find similar information quickly and efficiently.
4. Ask questions to ChatGPT and get answers along with the source URLs.

#### Installation
1.Clone this repository to your local machine using:

```bash
  git clone (https://github.com/SarthakNikhal/Article-Researcher-LLM.git
```
2.Navigate to the project directory:

3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_here
```

#### Usage/Examples
1. Start the Streamlit app by running:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, enter the URLs of the articles you want to analyze.
- Click "Process URLs" to load and process the data
- The system will split the text, create embedding vectors, and index them using FAISS to make searching faster.
- One can now ask a question and get the answer based on those news articles

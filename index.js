const venom = require('venom-bot');
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { RetrievalQAChain } = require("langchain/chains");
const { OpenAI } = require("@langchain/openai");



venom
  .create({
    session: 'session-name' //name of session
  })
  .then((client) => start(client))
  .catch((erro) => {
    console.log(erro);
  });

function start(client) {
  client.onMessage((message) => {
    if (message.body) {

     
    go(message.body,client,message.from);
    }
  });
}

async function go(question,client,message){

    const loader = new PDFLoader("src/ff.pdf");
    
    const docs = await loader.load();
    
    // splitter function
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 20,
    });
    
    // created chunks from pdf
    const splittedDocs = await splitter.splitDocuments(docs);
    
    const embeddings = new OpenAIEmbeddings({ openAIApiKey: "APIKEY"});
    
    const vectorStore = await HNSWLib.fromDocuments(
      splittedDocs,
      embeddings
    );
    const vectorStoreRetriever = vectorStore.asRetriever();
    const model = new OpenAI({
      modelName: 'gpt-3.5-turbo',
      openAIApiKey: "API_KEY",
    });
    
    const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
    
    
    
    const answer = await chain.call({
      query: question
    });
    const textPart = answer.text;
    console.log(textPart);
    client
          .sendText(message, textPart);
    
    
    
    }

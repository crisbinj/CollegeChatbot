import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config();

async function go(question){

const loader = new PDFLoader("src/ff.pdf");

const docs = await loader.load();

// splitter function
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 20,
});

// created chunks from pdf
const splittedDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings({ openAIApiKey: "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"});

const vectorStore = await HNSWLib.fromDocuments(
  splittedDocs,
  embeddings
);
const vectorStoreRetriever = vectorStore.asRetriever();
const model = new OpenAI({
  modelName: 'gpt-3.5-turbo',
  openAIApiKey: "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",
});

const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);



const answer = await chain.call({
  query: question
});
const textPart = answer.text;
console.log(textPart);



}

function first(){
    const question = 'what is mean by inflexibility in limitations of the model?';
    go(question);
}
first();
# Multi-RAG과 Multi-Region LLM으로 한국어 Chatbot 만들기 

[Amazon Bedrock](https://aws.amazon.com/ko/bedrock/)의 Anthropic Claude LLM(Large Language Models) 모델을 이용하여 질문/답변(Question/Answering)을 수행하는 Chatbot을 [Knowledge Database](https://aws.amazon.com/ko/about-aws/whats-new/2023/09/knowledge-base-amazon-bedrock-models-data-sources/)를 이용하여 구현합니다. 대량의 데이터로 사전학습(pretrained)한 대규모 언어 모델(LLM)은 학습되지 않은 질문에 대해서도 가장 가까운 답변을 맥락(context)에 맞게 찾아 답변할 수 있습니다. 또한, [RAG(Retrieval-Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)를 이용하면 LLM은 잘못된 답변(hallucination)의 영향을 줄일 수 있으며, [파인 튜닝(fine tuning)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html)을 제공하는 것처럼 최신의 데이터를 활용할 수 있습니다. RAG는 [prompt engineering](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-prompt-engineering.html) 기술 중의 하나로서 vector store를 지식 데이터베이스로 이용하고 있습니다. 

Vector store는 이미지, 문서(text document), 오디오와 같은 구조화 되지 않은 컨텐츠(unstructured content)를 저장하고 검색할 수 있습니다. 특히 대규모 언어 모델(LLM)의 경우에 embedding을 이용하여 텍스트들의 연관성(sementic meaning)을 벡터(vector)로 표현할 수 있으므로, 연관성 검색(sementic search)을 통해 질문에 가장 가까운 답변을 찾을 수 있습니다. 여기서는 대표적인 In-memory vector store인 [Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)와 persistent store이면서 대용량 병렬처리가 가능한 [Amazon OpenSearch](https://medium.com/@pandey.vikesh/rag-ing-success-guide-to-choose-the-right-components-for-your-rag-solution-on-aws-223b9d4c7280)와 완전관리형 검색서비스인 Kendra를 이용하여 RAG를 구현합니다.

Multiple RAG에서는 각 RAG가 관련 문서(Relevant Documents)를 생성하게 되므로, 전체 관련 문서들에서 실제 LLM에 전달할 문서의 숫자를 제한하여야 합니다. 이것은 Cluade2.1에서 200k token을 제공함에도 여전히 LLM의 context window에는 제한이 있기 때문입니다. 2023년 re:Invent에서는 다양한 Aurora, OpenSearch, Kendra뿐 아니라 Document DB, Mongo DB, Neptune등 거의 모든 데이터베이스의 RAG 지원이 발표되었습니다. 따라서, 향후 다양한 Database가 RAG의 소스로서 제공될 수 있습니다. 

Multiple LLM을 사용하게 되는 케이스에는 1) 다른 종류의 LLM을 사용하는 케이스 2) 같은 LLM을 다른 리전에서 사용하는 케이스가 있을 수 있습니다. LLM마다 잘하는 범위가 다를수 있으므로 다른 LLM의 결과를 통합해야할 수 있습니다. 또한, Bedrock On Demend의 경우에 단위시간당 보낼수 있는 Reuqest의 숫자 및 Token의 숫자가 제한되므로, 다른 리전의 동일 모델을 사용할 수 있다면, Provisioned로 사용하기 어려운 경우에 유용하게 이용될 수 있습니다. 1)의 경우에는 동시에 같은 요청을 보내서 다른 응답을 통합하는 과정이 필요하며, 2의 경우에는 Load를 분배하고 관리하는 방법이 필요합니다.

실행시간을 단축하기 위하여 Multi Thread를 사용합니다. [Lambda의 Multi thread](https://aws.amazon.com/ko/blogs/compute/parallel-processing-in-python-with-aws-lambda/)을 사용하므로써 연속적인 작업(sequencial job)을 병렬처리 할 수 있습니다.

[Lambda-Parallel Processing](https://aws.amazon.com/ko/blogs/compute/parallel-processing-in-python-with-aws-lambda/)와 같이 Lambda에서는 Pipe()을 이용합니다. 

## 아키텍처 개요

전체적인 아키텍처는 아래와 같습니다. 사용자는 Amazon CloudFront를 통해 [Amazon S3](https://aws.amazon.com/ko/s3/)로 부터 웹페이지에 필요한 리소스를 읽어옵니다. 사용자가 Chatbot 웹페이지에 로그인을 하면, 사용자 아이디를 이용하여 Amazon DynamoDB에 저장된 채팅 이력을 로드합니다. 이후 사용자가 메시지를 입력하면 WebSocket을 이용하여 LLM에 질의를 하게 되는데, DynamoDB로 부터 읽어온 채팅 이력과 RAG를 제공하는 Vector Database로부터 읽어온 관련문서(Relevant docs)를 이용하여 적절한 응답을 얻습니다. 이러한 RAG 구현을 위하여 [LangChain을 활용](https://python.langchain.com/docs/get_started/introduction.html)하여 Application을 개발하였고, Chatbot을 제공하는 인프라는 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/d4b20e2d-f392-438a-864c-b87d0bcd8822" width="800">

상세하게 단계별로 설명하면 아래와 같습니다.

단계1: 브라우저를 이용하여 사용자가 CloudFront 주소로 접속하면, Amazon S3에서 HTML, CSS, JS등의 파일을 전달합니다. 이때 로그인을 수행하고 채팅 화면으로 진입합니다.

단계2: Client는 사용자 아이디를 이용하여 '/history' API로 채팅이력을 요청합니다. 이 요청은 API Gateway를 거쳐서 lambda-history에 전달됩니다. 이후 DynamoDB에서 채팅 이력을 조회한 후에 다시 API Gateway와 lambda-history를 통해 사용자에게 전달합니다.

단계3: Client가 API Gateway로 WebSocket 연결을 시도하면, API Gateway를 거쳐서 lambda-chat-ws로 WebSocket connection event가 전달됩니다. 이후 사용자가 메시지를 입력하면, API Gateway를 거쳐서 lambda-chat-ws로 메시지가 전달됩니다.

단계4: lambda-chat-ws은 사용자 아이디를 이용하여 DynamoDB의 기존 채팅이력을 읽어와서, 채팅 메모리에 저장합니다.

단계5: lambda-chat-ws은 RAG에 관련된 문서(relevant docs)를 검색합니다.

단계6: lambda-chat-ws은 사용자의 질문(question), 채팅 이력(chat history), 관련 문서(relevant docs)를 Amazon Bedrock의 Enpoint로 전달합니다.

단계7: Amazon Bedrock의 사용자의 질문과 채팅이력이 전달되면, Anthropic의 Claude LLM을 이용하여 적절한 답변(answer)을 사용자에게 전달합니다. 이때, stream을 사용하여 답변이 완성되기 전에 답변(answer)를 사용자에게 보여줄 수 있습니다.

이때의 Sequence diagram은 아래와 같습니다.

![seq-chat](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/13818855-0a63-4d5e-9f9b-1b98245f80b6)


## 주요 구성

### Bedrock을 LangChain으로 연결

현재(2023년 9월) Bedrock의 상용으로 제한없이 AWS Bedrock을 사용할 수 있습니다. [Bedrock](https://python.langchain.com/docs/integrations/providers/bedrock)을 import하여 LangChain로 application을 개발할 수 있습니다. 여기서는 bedrock_region으로 us-east-1을 사용합니다. 

아래와 같이 bedrock client를 정의합니다. 서비스이름은 "bedrock-runtime"입니다.

```python
import boto3
from langchain.llms.bedrock import Bedrock

boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs=parameters)
```

여기서 파라메터는 아래와 같습니다.

```python
def get_parameter(modelId):
    if modelId == 'amazon.titan-tg1-large' or modelId == 'amazon.titan-tg1-xlarge': 
        return {
            "maxTokenCount":1024,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        }
    elif modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
        return {
            "max_tokens_to_sample":8191, # 8k
            "temperature":0.1,
            "top_k":250,
            "top_p": 0.9,
            "stop_sequences": [HUMAN_PROMPT]            
        }
parameters = get_parameter(modelId)
```

Bedrock의 지원모델은 [service name이 "bedrock"](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html)의 list_foundation_models()을 이용하여 조회합니다. 

```python
bedrock_client = boto3.client(
    service_name='bedrock',
    region_name=bedrock_region,
)
modelInfo = bedrock_client.list_foundation_models()    
print('models: ', modelInfo)
```



### Embedding

[BedrockEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/bedrock)을 이용하여 Embedding을 합니다. 'amazon.titan-embed-text-v1'은 Titan Embeddings Generation 1 (G1)을 의미하며 8k token을 지원합니다.

```python
bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' 
)
```

## Knowledge Database 정의

여기서는 Knowledge Database로 OpenSearch, Faiss, Kendra에 대해 알아봅니다.

### OpenSearch

[OpenSearchVectorSearch](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.opensearch_vector_search.OpenSearchVectorSearch.html)을 이용해 vector store를 정의합니다. 여기서 engine은 기본값이 nmslib이지만 필요에 따라 faiss나 lucene를 선택할 수 있습니다.

```python
from langchain.vectorstores import OpenSearchVectorSearch

vectorstore = OpenSearchVectorSearch(
    index_name = 'rag-index-'+userId+'-*',
    is_aoss = False,
    ef_search = 1024, # 512(default)
    m=48,
    #engine="faiss",  # default: nmslib
    embedding_function = bedrock_embeddings,
    opensearch_url=opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
)
```

OpenSearch를 이용한 vector store에 데이터는 아래와 같이 add_documents()로 넣을 수 있습니다. 여기서는 index를 이용해 개인화된 RAG를 적용하기 위하여 아래와 같이 index를 userId와 requestId로 정의한 후에 new vector store를 정의하여 이용합니다.

```python
new_vectorstore = OpenSearchVectorSearch(
    index_name="rag-index-"+userId+'-'+requestId,
    is_aoss = False,
    #engine="faiss",  # default: nmslib
    embedding_function = bedrock_embeddings,
    opensearch_url = opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd),
)
new_vectorstore.add_documents(docs)      
```

관련된 문서(relevant docs)는 아래처럼 검색할 수 있습니다.

```python
relevant_documents = vectorstore.similarity_search(query)
```

### Faiss

아래와 같이 Faiss를 vector store로 정의합니다. 여기서 Faiss는 in-memory vectore store로 인스턴스가 유지될 동안만 사용할 수 있습니다. 또한 faiss vector store에 데이터를 넣기 위해 add_documents()를 이용합니다. 데이터를 넣은 상태에서 검색을 할 수 있으므로, 아래와 같이 isReady를 체크합니다. 

```python
vectorstore = FAISS.from_documents( # create vectorstore from a document
    docs,  # documents
    bedrock_embeddings  # embeddings
)
isReady = True

vectorstore.add_documents(docs)
```

관련된 문서(relevant docs)는 아래처럼 검색할 수 있습니다.

```python
query_embedding = vectorstore.embedding_function(query)
relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
```

### Kendra

Kendra는 embedding이 필요하지 않으므로 아래와 같이 index_id를 설정하여 retriever를 지정합니다.

```python
from langchain.retrievers import AmazonKendraRetriever
kendraRetriever = AmazonKendraRetriever(index_id=kendraIndex)
```

[kendraRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.kendra.AmazonKendraRetriever.html?highlight=kendraretriever#langchain.retrievers.kendra.AmazonKendraRetriever)를 이용해 아래와 같이 관련된 문서를 검색할 수 있습니다.

```python
relevant_documents = kendraRetriever.get_relevant_documents(query)
```

### 관련된 문서를 포함한 RAG 구현

실제 결과는 [RetrievalQA](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html?highlight=retrievalqa#langchain.chains.retrieval_qa.base.RetrievalQA)을 이용해 얻습니다.

relevant_documents = vectorstore.similarity_search(query)

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
result = qa({"query": query})    
```

여기서 retriever는 아래와 같이 정의합니다. 여기서 kendra의 retriever는 [AmazonKendraRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.kendra.AmazonKendraRetriever.html?highlight=kendraretriever#langchain.retrievers.kendra.AmazonKendraRetriever)로 정의하고, opensearch와 faiss는 [VectorStore](https://api.python.langchain.com/en/latest/schema/langchain.schema.vectorstore.VectorStore.html?highlight=as_retriever#langchain.schema.vectorstore.VectorStore.as_retriever)을 이용합니다.

```python
if rag_type=='kendra':
    retriever = kendraRetriever
elif rag_type=='opensearch' or rag_type=='faiss':
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 3
        }
    )
```


### Reference 표시하기

아래와 같이 kendra는 doc의 metadata에서 reference에 대한 정보를 추출합니다. 여기서 file의 이름은 doc.metadata['title']을 이용하고, 페이지는 doc.metadata['document_attributes']['_excerpt_page_number']을 이용해 얻습니다. URL은 cloudfront의 url과 S3 bucket의 key, object를 조합하여 구성합니다. opensearch와 faiss는 파일명, page 숫자, 경로(URL path)를 metadata의 'name', 'page', 'url'을 통해 얻습니다.

```python
def get_reference(docs, rag_type):
    if rag_type == 'kendra':
        reference = "\n\nFrom\n"
        for doc in docs:
            name = doc.metadata['title']
            url = path+name

            if doc.metadata['document_attributes']:
                page = doc.metadata['document_attributes']['_excerpt_page_number']
                reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
            else:
                reference = reference + f"in <a href={url} target=_blank>{name}</a>\n"
    else:
        reference = "\n\nFrom\n"
        for doc in docs:
            name = doc.metadata['name']
            page = doc.metadata['page']
            url = doc.metadata['url']
        
            reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
        
    return reference
```


### Prompt의 생성

RAG와 대화 이력(chat history)를 모두 이용해 질문의 답변을 얻기 위해서는 [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html?highlight=conversationalretrievalchain#langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain)을 이용하거나, history와 현재의 prompt로 새로운 prompt를 생성한 이후에 retrivalQA를 이용해 얻을수 있습니다.

대화 이력(chat history)를 고려한 현재의 prompt를 생성하는 방법은 아래와 같습니다. 여기서는 prompt template에 "rephrase the follow up question"를 포함하여 새로운 질문을 생성합니다. 

```python
generated_prompt = get_generated_prompt(text)

def get_generated_prompt(query):    
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template = condense_template, input_variables = ["chat_history", "question"]
    )
    
    chat_history = extract_chat_history_from_memory()
    
    question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    return question_generator_chain.run({"question": query, "chat_history": chat_history})
```

이후, 생성된 질문과 RetrievalQA를 이용해 RAG 적용한 결과를 얻을 수 있습니다.

## 메모리에 대화 저장

### RAG를 사용하지 않는 경우

lambda-chat-ws는 인입된 메시지의 userId를 이용하여 map_chat에 기저장된 대화 이력(memory_chat)가 있는지 확인합니다. 채팅 이력이 없다면 아래와 같이 [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory)로 memory_chat을 설정합니다. 여기서, Anhropic Claude는 human과 ai의 이름으로 "Human"과 "Assistant"로 설정합니다. LLM에 응답을 요청할때에는 ConversationChain을 이용합니다.

```python
map_chat = dict()

if userId in map_chat:  
    memory_chat = map_chat[userId]
else:
    memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
    map_chat[userId] = memory_chat
conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chat)
```

[ConversationBufferWindowMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)을 이용하여 간단하게 k개로 conversation의 숫자를 제한할 수 있습니다.


여기서는 Faiss를 이용할때 대화이력이 없는 경우에는 RAG를 쓸수 없으므로 위와 같이 적용합니다.

### RAG를 사용하는 경우

RAG를 이용할때는 [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory)을 이용해 아래와 같이 채팅 메모리를 지정합니다. 대화가 끝난후에는 add_user_message()와 add_ai_message()를 이용하여 새로운 chat diaglog를 업데이트 합니다.

```python
map_chain = dict()

if userId in map_chain:  
    memory_chain = map_chain[userId]
else: 
    memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    map_chain[userId] = memory_chain

msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)      

memory_chain.chat_memory.add_user_message(text)  # append new diaglog
memory_chain.chat_memory.add_ai_message(msg)

def get_answer_from_conversation(text, conversation, convType, connectionId, requestId):
    conversation.prompt = get_prompt_template(text, convType)
    stream = conversation.predict(input=text)                        
    msg = readStreamMsg(connectionId, requestId, stream)

    return msg
```

여기서 stream은 아래와 같은 방식으로 WebSocket을 사용하는 client에 메시지를 전달할 수 있습니다.

```python
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg
            }
            sendMessage(connectionId, result)
    print('msg: ', msg)
    return msg
```

여기서 client로 메시지를 보내는 sendMessage()는 아래와 같습니다. 여기서는 boto3의 [post_to_connection](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigatewaymanagementapi/client/post_to_connection.html)를 이용하여 메시지를 WebSocket의 endpoint인 API Gateway로 전송합니다.

```python
def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except: 
        raise Exception ("Not able to send a message")
```

### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


### 실행결과

[fsi_faq_ko.csv](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/fsi_faq_ko.csv)을 다운로드한 후에 파일 아이콘을 선택하여 업로드하면 Knowledge Database에 저장됩니다. 이후 아래와 같이 파일 내용을 확인할 수 있도록 요약하여 보여줍니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a0b3b5b8-6e1e-4240-9ee4-e539680fa28d)

채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. 이때의 결과는 ＂아니오”입니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c7aeca05-0209-49c3-9df9-7e04026900f2)

채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/56ad9192-6b7c-49c7-9289-b6a3685cb7d4)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?” 라고 입력하면 아래와 같은 결과를 얻을 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/95a78e6a-5a78-4879-98a1-a30aa6f7e3d5)



아래와 같이 채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/849169c3-c9ee-40dd-8e44-0ec308688ce6)


채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. "영문뱅킹에서는 간편조회서비스 이용불가"하므로 좀더 자세한 설명을 얻었습니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/3a896488-af0c-42b2-811b-d2c0debf5462)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?"라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/2e2b2ae1-7c50-4c14-968a-6c58332d99af)




#### Translation

"아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다.”로 입력하고 번역 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/818662e1-983f-44c2-bfcf-e2605ba7a1e6)

#### Extracted Topic and sentiment

“식사 가성비 좋습니다. 위치가 좋고 스카이라운지 바베큐 / 야경 최곱니다. 아쉬웠던 점 · 지하주차장이 비좁습니다.. 호텔앞 교통이 너무 복잡해서 주변시설을 이용하기 어렵습니다. / 한강나가는 길 / 주변시설에 나가는 방법등.. 필요합니다.”를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c38a58b-08df-4e9e-a162-1cd8f542fb46)

#### Information extraction

“John Park. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-5555“로 입력후에 이메일이 추출되는지 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/f613e86e-b08d-45e8-ac0e-71334427f450)

#### PII(personally identifiable information) 삭제하기

PII(Personal Identification Information)의 삭제의 예는 아래와 같습니다. "John Park, Ph.D. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-4567"와 같이 입력하여 name, phone number, address를 삭제한 텍스트를 얻습니다. 프롬프트는 [PII](https://docs.anthropic.com/claude/docs/constructing-a-prompt)를 참조합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a77d034c-32fc-4c84-8054-f4e1230292d6)

#### 문장 오류 고치기

"To have a smoth conversation with a chatbot, it is better for usabilities to show responsesess in a stream-like, conversational maner rather than waiting until the complete answer."로 오류가 있는 문장을 입력합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/55774e11-58e3-4eb4-b91c-5b09572456bd)

"Chatbot과 원할한 데화를 위해서는 사용자의 질문엥 대한 답변을 완전히 얻을 때까지 기다리기 보다는 Stream 형태로 보여주는 것이 좋습니다."로 입력후에 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7b098a29-9bf5-43bf-a32f-82c94ccd04eb)

#### 복잡한 질문 (step-by-step)

"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c1bf6749-1ce8-44ba-81f1-1fb52e04a2e8)


"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까?"로 질문을 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/992c8385-f897-4411-b6cf-b185465e8690)

#### 날짜/시간 추출하기

메뉴에서 "Timestamp Extraction"을 선택하고, "지금은 2023년 12월 5일 18시 26분이야"라고 입력하면 prompt를 이용해 아래처럼 시간을 추출합니다.

![noname](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7dd7e659-498c-4898-801c-b72830bf254b)


실제 결과 메시지는 아래와 같습니다. 

```xml
<result>
<year>2023</year>
<month>12</month>
<day>05</day>
<hour>18</hour>
<minute>26</minute>
</result>
```

#### 어린이와 대화 (Few shot example)

대화의 상대에 맞추어서 질문에 답변을하여야 합니다. 이를테면 [General Conversation]에서 "산타가 크리스마스에 선물을 가져다 줄까?"로 질문을 하면 아래와 같이 답변합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4624727b-addc-4f5d-8f3d-94f358572326)

[9. Child Conversation (few shot)]으로 전환합니다. 동일한 질문을 합니다. 상대에 맞추어서 적절한 답변을 할 수 있었습니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/cbbece5c-5476-4f3b-89f7-c7fcf90ca796)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "rest-api-for-stream-chatbot", "ws-api-for-stream-chatbot"을 삭제합니다.

2) [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cdk destroy --all
```

## 문제점

Retrive는 [ScoreAttributes](https://docs.aws.amazon.com/kendra/latest/APIReference/API_ScoreAttributes.html)는 한국어에 대해 항상 "NOT_AVAILABLE" 이므로, 전혀 관련없는 문서들이 RAG에 검색이 될 수 있습니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/2b0f90b9-f584-4274-a42c-cf9f196bcac7)


## 결론

AWS 서울 리전에서 Amazon Bedrock과 vector store를 이용하여 질문과 답변(Question/Answering)을 수행하는 chatbot을 구현하였습니다. Amazon Bedrock은 여러 종류의 대용량 언어 모델중에 한개를 선택하여 사용할 수 있습니다. 여기서는 Amazon Titan을 이용하여 RAG 동작을 구현하였고, 대용량 언어 모델의 환각(hallucination) 문제를 해결할 수 있었습니다. 또한 Chatbot 어플리케이션 개발을 위해 LangChain을 활용하였고, IaC(Infrastructure as Code)로 AWS CDK를 이용하였습니다. 대용량 언어 모델은 향후 다양한 어플리케이션에서 효과적으로 활용될것으로 기대됩니다. Amazon Bedrock을 이용하여 대용량 언어 모델을 개발하면 기존 AWS 인프라와 손쉽게 연동하고 다양한 어플리케이션을 효과적으로 개발할 수 있습니다.



## Reference 

[Claude - Constructing a prompt](https://docs.anthropic.com/claude/docs/constructing-a-prompt)

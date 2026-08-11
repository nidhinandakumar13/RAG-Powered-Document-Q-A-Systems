import pdfplumber #these are tookits. pfdplumber knows how to read a pdf file and can pull text
import io #built in for working with data streams
from openai import OpenAI #lets us talk to AI
from dotenv import load_dotenv #will read the .env file
import chromadb #hold the chunkc
import streamlit as st

load_dotenv()
client = OpenAI()
chroma = chromadb.Client() #starts a connection to a vector
collection = chroma.get_or_create_collection("documents") #creates a bucket inside the database called documents where we will keep the chunks and their embeddings


def extract_text_from_pdf(file_bytes: bytes): #this def will take a pdf file which is raw(0 and 1) this will return a list of string
    chunks = [] #this is an empty list that we will fill with pieces of text
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf: # that long .open part takes the raw data from bytes and then makes it into an object that acts like an open file so it cane be read as .pfd
        for page in pdf.pages:  #loops through each page
            text = page.extract_text() #for each page, ask the pdf plumber to read all the text in the pdf as one string
            if text: #this is here so if the pdf scanner seen a picture or the pdf is just a pictire it wont crash
                paragraph = [p.strip() for p in text.split('\n\n') if p.strip()] #checks to see if there is two new lines, which will mean there is a paraghaph
                chunks.extend(paragraph) #adds all the paragraphs from this page into a chunk list, the .extend adds each item individually
    return chunks

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input = text) #turns the text into numbers so 
    return response.data[0].embedding #similar meaning will end up with similar numbers


def stores_chunks(chunks):
    old_ids = collection.get()["ids"] # Remove chunks from the previous PDF

    if old_ids:
        collection.delete(ids=old_ids)
    embeddings = [get_embedding(chunk) for chunk in chunks] #this is a list that goes thru every chunk in ur list and calls the get_embedding function

    ids = [f"chunk_{i}" for i in range(len(chunks))] #ChromaDB is like a filing cabinate. ur assigning a chunk to unqiue label
    collection.add(     # Store only the new PDF

        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

def ask_question(question): # takes the question from the user
    q_embedding = get_embedding(question) #we are turning every question into number thing we did earlier

    results = collection.query(query_embeddings = [q_embedding], n_results=3) #the .query is looking through everything we stories in Chroma and the [] is looking at wht we are searching for. the n=3 is saying give me back 3 chunks that are closest to this answer
    context = "\n\n".join(results["documents"][0])

    response = client.chat.completions.create(   #connecting to chat 
        model="gpt-4o-mini",
        messages=[{
            "role":"user",
            "content": f"""Answer using ONLY this context:
{context}

Question: {question}""" 
        }]
    )
    return response.choices[0].message.content



#if __name__=="__main__":
    #with open("test.pdf","rb") as f: #opens the file and Reads Binary and then the F is for reading the while thing into memory and storing it int file_bytes
        #file_bytes = f.read()

        #result = extract_text_from_pdf(file_bytes)
        #print(f"Found {len(result)} chunks")

        #stores_chunks(result)
        #print("Stored chunks in ChromaDB!")
        #print(f"Total items in collection: {collection.count()}")

        #answer = ask_question("Who is the professor")
        #print("Answer:", answer)

        #first_embedding = get_embedding(result[0])
        #print(f"Embedding length: {len(first_embedding)}")
        #print(f"First few numbers: {first_embedding[:5]}")
        #print("First chunk:" , result[0] if result else "No chunks found")'''

st.title("RAG Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    chunks = extract_text_from_pdf(file_bytes)

    stores_chunks(chunks)

    st.success("PDF uploaded successfully!")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        answer = ask_question(question)

        st.write("Answer:")
        st.write(answer)


import os
import pandas as pd
import numpy as np
import dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pickle
import uvicorn
import logging
from typing import Dict, List
import uuid

# Install: pip install implicit
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares


# Import Langchain, Qdrant, and LLM related components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
vectorstore = None
model = None
user_id_map = None
product_id_map = None
reverse_user_id_map = None
reverse_product_id_map = None
sparse_matrix = None
tools = None

# Conversation sessions storage
conversation_sessions: Dict[str, ConversationBufferMemory] = {}

# Initialize FastAPI app
app = FastAPI(title="Product Recommendation System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
def load_products_data():
    """
    Read CSV data, join products with aisles and departments,
    and generate a text description field "doc"
    """
    data_path = "data"
    products_path = os.path.join(data_path, "products.csv")
    aisles_path = os.path.join(data_path, "aisles.csv")
    departments_path = os.path.join(data_path, "departments.csv")
    
    logger.info(f"Loading data from {products_path}, {aisles_path}, and {departments_path}")
    
    try:
        products = pd.read_csv(products_path)
        aisles = pd.read_csv(aisles_path)
        departments = pd.read_csv(departments_path)
        
        # Merge based on CSV field names (ensure primary key field names are correct)
        products = products.merge(aisles, on="aisle_id", how="left")
        products = products.merge(departments, on="department_id", how="left")
        
        # Generate text description, e.g.:
        # "Product name: xxx; Aisle: xxx; Department: xxx"
        def create_doc(row):
            return f"product name：{row['product_name']}；aisle：{row.get('aisle', '')}；department：{row.get('department', '')}"
        
        products["doc"] = products.apply(create_doc, axis=1)
        logger.info(f"Successfully loaded {len(products)} products")
        return products
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def init_vector_store():
    """
    Create a new vector store with product data
    This function assumes the collection doesn't exist or has been deleted
    """
    products = load_products_data()
    texts = products["doc"].tolist()
    
    logger.info(f"Creating new vector store with {len(texts)} documents")
    
    try:
        # Ensure OpenAIEmbeddings is initialized correctly
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Connect to Qdrant service
        client = QdrantClient(host="qdrant", port=6333)
        collection_name = "products_collection"
        
        # Create collection (assumes it doesn't exist)
        vector_size = 1536  # OpenAI embeddings dimension
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
        logger.info(f"Created collection: {collection_name}")
        
        # Create documents with metadata
        documents = []
        for _, row in products.iterrows():
            metadata = {
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'aisle': row.get('aisle', ''),
                'department': row.get('department', ''),
                'aisle_id': row['aisle_id'],
                'department_id': row['department_id']
            }
            documents.append(Document(page_content=row['doc'], metadata=metadata))
        
        # Use from_documents method to create vector store
        vector_store = QdrantVectorStore.from_documents(
            documents,
            embeddings,
            url=f"http://qdrant:6333",
            collection_name=collection_name,
        )
        
        logger.info(f"Successfully created vector store with {len(texts)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

# ------------------------------------------------------------
# Train ALS model
# ------------------------------------------------------------
def train_als_model():
    """
    Load user-product purchase counts and train ALS model
    Skip training if pickle file already exists
    """
    # Check if ALS model pickle file already exists
    if os.path.exists('als_model.pkl'):
        logger.info("ALS model pickle file found, skipping training")
        return None, None, None, None, None, None
    
    data_path = "data"
    orders_path = os.path.join(data_path, "orders.csv")
    orders_products_prior_path = os.path.join(data_path, "order_products__prior.csv.gz")
    
    logger.info(f"Loading data from {orders_path} and {orders_products_prior_path}")
    
    try:
        orders = pd.read_csv(orders_path)
        order_products_prior = pd.read_csv(orders_products_prior_path)
        
        user_product = (
            orders[['order_id', 'user_id']]
            .merge(order_products_prior[['order_id', 'product_id']], on='order_id')
            .groupby(['user_id', 'product_id'])
            .size()
            .reset_index(name='times_purchased')
        )
        # Create a sparse matrix: rows=users, cols=products, values=times_purchased
        user_ids = user_product['user_id'].astype('category').cat.codes
        product_ids = user_product['product_id'].astype('category').cat.codes

        sparse_matrix = sparse.coo_matrix(
            (user_product['times_purchased'], (user_ids, product_ids))
        )

        # Train ALS model
        model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)
        model.fit(sparse_matrix)

        # map between your original user/product IDs and the integer indices used in your ALS model's matrix.
        ## user_id_map[0] gives you the real user ID for matrix row 0.
        ## product_id_map[0] gives you the real product ID for matrix column 0.
        ## reverse_user_id_map[real_user_id] gives you the matrix row index for a given user.
        ## reverse_product_id_map[real_product_id] gives you the matrix column index for a given product.

        user_id_map = dict(enumerate(user_product['user_id'].astype('category').cat.categories))
        product_id_map = dict(enumerate(user_product['product_id'].astype('category').cat.categories))
        reverse_user_id_map = {v: k for k, v in user_id_map.items()}
        reverse_product_id_map = {v: k for k, v in product_id_map.items()}

        # Save ALS model and mappings
        als_data = {
            'model': model,
            'user_id_map': user_id_map,
            'product_id_map': product_id_map,
            'reverse_user_id_map': reverse_user_id_map,
            'reverse_product_id_map': reverse_product_id_map,
            'sparse_matrix': sparse_matrix
        }
        
        with open('als_model.pkl', 'wb') as f:
            pickle.dump(als_data, f)
        
        logger.info("✅ Saved ALS model and mappings to pickle file")

        return model, user_product, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map

    except Exception as e:
        logger.error(f"Error training ALS model: {e}")
        raise

def load_als_vectorstore():
    """Load ALS model from saved files and initialize vector store efficiently"""
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Connect to Qdrant and check if collection exists
    client = QdrantClient(host="qdrant", port=6333)
    collection_name = "products_collection"
    
    try:
        # Check if collection exists and has data
        collection_info = client.get_collection(collection_name=collection_name)
        collection_points = client.count(collection_name=collection_name)
        
        if collection_points.count > 0:
            logger.info(f"✅ Collection {collection_name} exists with {collection_points.count} vectors, using existing vector store")
            
            # Return existing vector store without reprocessing data
            vectorstore = QdrantVectorStore.from_existing_collection(
                collection_name=collection_name,
                embedding=embeddings,
                host="qdrant",
                port=6333
            )
        else:
            logger.info(f"⚠️ Collection {collection_name} exists but is empty, recreating vector store")
            # Delete empty collection and recreate
            client.delete_collection(collection_name=collection_name)
            vectorstore = init_vector_store()
        
    except Exception as e:
        # Collection doesn't exist, create it
        logger.info(f"🔄 Collection {collection_name} not found, creating new vector store")
        vectorstore = init_vector_store()
    
    # Check if ALS model pickle exists
    if not os.path.exists('als_model.pkl'):
        logger.error("ALS model pickle file not found. Please run train_als_model() first.")
        raise FileNotFoundError("als_model.pkl not found")
    
    # Load ALS model and mappings
    try:
        with open('als_model.pkl', 'rb') as f:
            als_data = pickle.load(f)
        
        # Extract components
        model = als_data['model']
        user_id_map = als_data['user_id_map']
        product_id_map = als_data['product_id_map']
        reverse_user_id_map = als_data['reverse_user_id_map']
        reverse_product_id_map = als_data['reverse_product_id_map']
        sparse_matrix = als_data['sparse_matrix']
        
        logger.info("✅ Loaded ALS model and vector store successfully")
        return vectorstore, model, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map, sparse_matrix
        
    except Exception as e:
        logger.error(f"Error loading ALS model: {e}")
        raise



# Calls the ALS model’s recommend method to get the top-N recommended products for this user.
def recommend_for_user(user_id, N=10):
    user_idx = reverse_user_id_map[user_id]
    
    user_row = sparse_matrix.tocsr()[user_idx]
    recommended = model.recommend(user_idx, user_row, N=N)
    # print(recommended)
    # recommended is a list of (product_idx, score)
    item_indices, scores = recommended

    return [(product_id_map[pid], float(score)) for pid, score in zip(item_indices, scores)]


def create_enhanced_tools(vectorstore, user_id=None):
    """Create enhanced tools with better integration"""
    
    def get_recommendations(user_id: str, top_n: int = 5) -> str:
        """Get product recommendations for a specific user - returns product details"""
        try:
            # Clean the user_id string - remove quotes and extra whitespace
            if isinstance(user_id, str):
                user_id = user_id.strip().strip("'\"")
            
            # Convert string to integer
            user_id_int = int(user_id)
            
            # Get more recommendations than requested to ensure we have enough unique products
            recs = recommend_for_user(user_id_int, N=top_n * 2)
            
            # Remove duplicates and get unique products
            seen_products = set()
            unique_recs = []
            
            for product_id, score in recs:
                if product_id not in seen_products and len(unique_recs) < top_n:
                    seen_products.add(product_id)
                    unique_recs.append((product_id, score))
            
            if not unique_recs:
                return f"No unique product recommendations found for user {user_id_int}."
            
            # Get product details for each recommendation
            result = f"Here are the top {len(unique_recs)} product recommendations for user {user_id_int}:\n\n"
            for i, (product_id, score) in enumerate(unique_recs, 1):
                # Get product details
                product_details = get_product_details(str(product_id))
                result += f"{i}. {product_details}; **Score:** {score:.3f}\n\n"
            
            return result
        except ValueError:
            return f"Error: User ID '{user_id}' is not a valid integer."
        except KeyError:
            return f"Error: User {user_id} not found in the recommendation system."
        except Exception as e:
            return f"Error getting recommendations: {str(e)}"

    def get_recommendations_with_count(user_id: str, count: str = "5") -> str:
        """Get product recommendations with specified count"""
        try:
            # Handle comma-separated input (e.g., "1256, 6")
            if "," in user_id:
                parts = user_id.split(",")
                if len(parts) >= 2:
                    user_id = parts[0].strip()
                    count = parts[1].strip()
            
            # Parse the count parameter
            top_n = int(count)
            return get_recommendations(user_id, top_n)
        except ValueError:
            return f"Error: Invalid count '{count}'. Please provide a valid number."
        except Exception as e:
            return f"Error getting recommendations: {str(e)}"

    def enhanced_search_tool(query: str, top_k: int = 5) -> str:
        """Enhanced product search with better formatting"""
        try:
            # Use the global vectorstore instead of loading FAISS
            docs = vectorstore.similarity_search(query, k=top_k)
            
            if not docs:
                return f"No products found matching: {query}"
            
            result = f"Found {len(docs)} products related to '{query}':\n\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                # Format exactly as requested
                result += f"{i}. **Product:** {metadata.get('product_name', 'Unknown Product')}; **Aisle:** {metadata.get('aisle', 'Unknown')}; **Department:** {metadata.get('department', 'Unknown')}\n\n"
            
            return result
        except Exception as e:
            return f"Error searching products: {str(e)}"
    
    def get_product_details(product_id: str) -> str:
        """Get detailed information about a specific product by ID"""
        try:
            product_id_int = int(product_id)
            
            # Create a search query for this specific product
            search_query = f"product_id:{product_id_int}"
            docs = vectorstore.similarity_search(search_query, k=1)
            
            if docs:
                metadata = docs[0].metadata
                # Format exactly as requested
                result = f"**Product:** {metadata.get('product_name', 'Unknown Product')}; **Aisle:** {metadata.get('aisle', 'Unknown')}; **Department:** {metadata.get('department', 'Unknown')}"
                return result
            else:
                return f"Product ID {product_id_int} not found in database."
                
        except ValueError:
            return f"Error: Product ID '{product_id}' is not a valid integer."
        except Exception as e:
            return f"Error getting product details: {str(e)}"
    
    # Create tools
    recommendation_tool = Tool(
        name="get_product_recommendations",
        description=f"""Get personalized product recommendations for user {user_id if user_id else '[USER_ID]'}. 
        
        Format output as: **Product:** [name]; **Aisle:** [aisle]; **Department:** [department]; **Score:** [score]
        
        Use this when users ask for recommendations, suggestions, or what they should buy. 
        Pass the user ID '{user_id}' as the first parameter and the number of recommendations as the second parameter (e.g., '10' for 10 products).""",
        func=get_recommendations_with_count
    )
    
    search_tool = Tool(
        name="search_product_database",
        description="""Search the product database for information about products, categories, departments, or general product queries. 
        
        Format output as: **Product:** [name]; **Aisle:** [aisle]; **Department:** [department]
        
        Use this when users ask about specific products, categories, or general product information.""",
        func=enhanced_search_tool
    )
    
    product_details_tool = Tool(
        name="get_product_details",
        description="""Get detailed information about a specific product by its ID. 
        
        Format output as: **Product:** [name]; **Aisle:** [aisle]; **Department:** [department]
        
        Use this to get product names, departments, and aisles for specific product IDs.""",
        func=get_product_details
    )
    
    return [recommendation_tool, search_tool, product_details_tool]


@app.on_event("startup")
async def startup_event():
    """
    Initialize the vector store, ALS model, and tools
    """
    global vectorstore, model, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map, sparse_matrix

    try:
        logger.info("🚀 Starting system initialization...")
        
        # Check if ALS model pickle file exists
        if os.path.exists('als_model.pkl'):
            logger.info("✅ ALS model pickle file found")
        else:
            logger.info("🔄 ALS model pickle not found, will train new model")
        
        # Try to train ALS model (will skip if pickle exists)
        train_result = train_als_model()
        
        # Load ALS model and vector store
        logger.info("📚 Loading ALS model and vector store...")
        vectorstore, model, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map, sparse_matrix = load_als_vectorstore()
        
        logger.info("✅ Startup event completed successfully.")
        
    except Exception as e:
        logger.error(f"❌ Fatal error initializing QA chain: {e}")
        # Terminate app on critical error
        import sys
        sys.exit(1)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """
    Return frontend home page
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify system initialization
    """
    if vectorstore is None:
        return JSONResponse({
            "status": "initializing",
            "message": "System is still starting up"
        })
    
    return JSONResponse({
        "status": "healthy",
        "message": "System is ready",
        "vectorstore_initialized": vectorstore is not None
    })

@app.post("/retrain")
async def retrain_models():
    """
    Manually trigger retraining of ALS model and recreation of vector store
    """
    try:
        # Remove existing pickle file to force retraining
        if os.path.exists('als_model.pkl'):
            os.remove('als_model.pkl')
            logger.info("Removed existing ALS model pickle file")
        
        # Force retrain ALS model
        train_als_model()
        
        # Reload everything
        global vectorstore, model, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map, sparse_matrix
        
        vectorstore, model, user_id_map, product_id_map, reverse_user_id_map, reverse_product_id_map, sparse_matrix = load_als_vectorstore()
        
        return JSONResponse({
            "status": "success",
            "message": "Models retrained successfully"
        })
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Error retraining models: {str(e)}"
        })


def create_hybrid_agent(session_id: str = None, user_id: str = None):
    """Create a hybrid agent with conversation memory"""
    global vectorstore, conversation_sessions
    
    # Initialize LLM with system message
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Get or create conversation memory for this session
    if session_id and session_id in conversation_sessions:
        memory = conversation_sessions[session_id]
    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        if session_id:
            conversation_sessions[session_id] = memory
    
    # Create tools dynamically with user_id context
    tools = create_enhanced_tools(vectorstore, user_id)
    
    # Create the agent with format instructions
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory,
        agent_kwargs={
            "system_message": """You are a product recommendation assistant. 
            
            IMPORTANT: When providing product information, always use consistent markdown formatting:
            
            **Product:** [product_name]  
            **Aisle:** [aisle]  
            **Department:** [department]
            
            For recommendations, include the reason:
            **Product:** [product_name]  
            **Aisle:** [aisle]  
            **Department:** [department]  
            **Score:** [score/explanation]
            
            Always provide clear, formatted responses with product details using bold markdown headers."""
        }
    )
    
    return agent

    
@app.post("/clear_history")
async def clear_conversation_history(request: Request):
    """Clear conversation history for a session"""
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    
    global conversation_sessions
    
    if session_id in conversation_sessions:
        # Clear the memory for this session
        conversation_sessions[session_id].clear()
        logger.info(f"Cleared conversation history for session: {session_id}")
        
        return JSONResponse({
            "message": "Conversation history cleared successfully",
            "session_id": session_id
        })
    else:
        return JSONResponse({
            "message": "Session not found",
            "session_id": session_id
        })


@app.post("/chat")
async def hybrid_agent_chat(request: Request):
    """Chat endpoint using hybrid agent with conversation memory"""
    data = await request.json()
    query = data.get("query")
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Check if tools are initialized
    if vectorstore is None:
        return JSONResponse({
            "error": "System is still initializing. Please try again in a moment.",
            "type": "error"
        })

    try:
        # Pre-check for greetings and recommendation requests to provide direct responses
        query_lower = query.lower()
        
        # Handle greetings directly
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            if user_id:
                response = f"Hello! I'm your product assistant. I can help you with product information and personalized recommendations. Your user ID is {user_id}, so I can provide personalized recommendations for you. Ask me 'What should I buy?' for personalized recommendations or ask about specific products and categories."
            else:
                response = "Hello! I'm your product assistant. I can help you with product information and personalized recommendations. Please enter your user ID above for personalized recommendations, or ask me about specific products and categories."
            
            return JSONResponse({
                "answer": response,
                "type": "direct_response",
                "session_id": session_id
            })
        
        # Handle recommendation requests without user ID directly
        if any(word in query_lower for word in ["recommend", "suggest", "buy", "should"]) and not user_id:
            response = "I'd be happy to provide personalized recommendations! Please enter your user ID in the field above and try again. For example, enter '123' in the User ID field, then ask 'What should I buy?'"
            
            return JSONResponse({
                "answer": response,
                "type": "direct_response",
                "session_id": session_id
            })
        
        # For general product queries (like "suggest some products"), let the agent handle it
        # Create hybrid agent with session memory and user ID context
        agent = create_hybrid_agent(session_id, user_id)
        
        # Run the agent with timeout
        response = agent.run(query)
        
        return JSONResponse({
            "answer": response,
            "type": "hybrid_agent_response",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        
        # Provide a helpful fallback response based on the query
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            if user_id:
                fallback_response = f"Hello! I'm your product assistant. I can help you with product information and personalized recommendations. Your user ID is {user_id}, so I can provide personalized recommendations for you. Ask me 'What should I buy?' for personalized recommendations or ask about specific products and categories."
            else:
                fallback_response = "Hello! I'm your product assistant. I can help you with product information and personalized recommendations. Please enter your user ID above for personalized recommendations, or ask me about specific products and categories."
        elif any(word in query_lower for word in ["recommend", "suggest", "buy", "should"]):
            if not user_id:
                # User is asking for recommendations without user ID
                fallback_response = "I'd be happy to provide personalized recommendations! Please enter your user ID in the field above and try again. For example, enter '123' in the User ID field, then ask 'What should I buy?'"
            else:
                fallback_response = "I'm here to help with product recommendations and information! You can ask me about specific products, categories, or request personalized recommendations. For example, try asking 'Tell me about organic fruits' or 'What should I buy?'"
        elif any(word in query_lower for word in ["product", "products", "find", "search"]):
            fallback_response = "I can help you find products! Try asking about specific categories like 'Tell me about soft drinks' or 'What fruits do you have?'"
        else:
            if user_id:
                fallback_response = f"I'm here to help you with product information and recommendations! Your user ID is {user_id}, so I can provide personalized recommendations. Ask me 'What should I buy?' for personalized recommendations or ask about specific products and categories."
            else:
                fallback_response = "I'm here to help you with product information and recommendations! Please enter your user ID above for personalized recommendations, or ask me about specific products and categories."
        
        return JSONResponse({
            "answer": fallback_response,
            "type": "fallback_response",
            "error": str(e),
            "session_id": session_id
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
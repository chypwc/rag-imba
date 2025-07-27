// Global session ID for conversation memory
let sessionId = null;

// Show welcome message when page loads
document.addEventListener("DOMContentLoaded", function () {
  appendMessage(
    "bot",
    "Hello! I'm your product assistant. I can help you with product information and personalized recommendations. To get personalized recommendations based on your purchase history, please enter your user ID in the field above. You can also ask me about specific products and categories without a user ID."
  );
});

// Listen for Enter key in input box
document.getElementById("query").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});

async function sendMessage() {
  const queryInput = document.getElementById("query");
  const userIdInput = document.getElementById("user-id");
  const query = queryInput.value;
  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";

  try {
    const requestBody = { query: query };
    
    // Include session ID if we have one
    if (sessionId) {
      requestBody.session_id = sessionId;
    }
    
    // Include user ID if provided
    const userId = userIdInput.value;
    if (userId) {
      requestBody.user_id = userId;
    }

    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    // Store session ID from response
    if (data.session_id && !sessionId) {
      sessionId = data.session_id;
      console.log("Session started with ID:", sessionId);
    }

    // Process response, try to extract product information and display as cards
    processResponse(data.answer);
  } catch (error) {
    appendMessage("bot", "Sorry, an error occurred: " + error.message);
  }
}

async function clearHistory() {
  if (!sessionId) {
    appendMessage("bot", "No active session to clear.");
    return;
  }

  try {
    const response = await fetch("/clear_history", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    
    // Clear the chat display
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = "";
    
    // Show welcome message again
    appendMessage(
      "bot",
      "Hello! I'm your product assistant. I can help you with product information and personalized recommendations. To get personalized recommendations based on your purchase history, please enter your user ID in the field above. You can also ask me about specific products and categories without a user ID."
    );
    
    appendMessage("bot", "Conversation history cleared successfully.");
    
  } catch (error) {
    appendMessage("bot", "Sorry, an error occurred while clearing history: " + error.message);
  }
}

function appendMessage(sender, text) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message " + sender;

  // Create sender label
  const senderSpan = document.createElement("span");
  senderSpan.innerText = sender === "user" ? "User: " : "System: ";
  senderSpan.style.fontWeight = "bold";

  msgDiv.appendChild(senderSpan);

  // Add message content with markdown support for bot messages
  const contentSpan = document.createElement("span");
  if (sender === "bot") {
    // Render markdown for bot messages
    contentSpan.innerHTML = marked.parse(text);
  } else {
    // Plain text for user messages
    contentSpan.innerText = text;
  }
  msgDiv.appendChild(contentSpan);

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function processResponse(answer) {
  // First check if it contains product information
  const productRegex =
    /Product Name[:：]([^；]+)；Aisle[:：]([^；]+)；Department[:：]([^；]+)/g;
  let match;
  let products = [];

  // Extract all product information
  while ((match = productRegex.exec(answer)) !== null) {
    products.push({
      name: match[1].trim(),
      aisle: match[2].trim(),
      department: match[3].trim(),
    });
  }

  // If product information is found, display as cards
  if (products.length > 0) {
    displayProductCards(products, answer);
  } else {
    // If no product information, display text response directly
    appendMessage("bot", answer);
  }
}

function displayProductCards(products, originalAnswer) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message bot";

  // Create sender label
  const senderSpan = document.createElement("span");
  senderSpan.innerText = "System: ";
  senderSpan.style.fontWeight = "bold";
  msgDiv.appendChild(senderSpan);

  // Extract non-product information part from system response
  let introText = originalAnswer;
  products.forEach((product) => {
    introText = introText.replace(
      `Product Name：${product.name}；Aisle：${product.aisle}；Department：${product.department}`,
      ""
    );
  });

  // Clean text, remove extra spaces and punctuation
  introText = introText
    .replace(/；+/g, "；")
    .replace(/；\s*；/g, "；")
    .replace(/；\s*$/g, "")
    .trim();

  if (introText) {
    const textDiv = document.createElement("div");
    textDiv.innerHTML = marked.parse(introText);
    msgDiv.appendChild(textDiv);
  }

  // Create product cards container
  const cardsContainer = document.createElement("div");
  cardsContainer.className = "product-cards";

  // Add product cards
  products.forEach((product) => {
    const card = document.createElement("div");
    card.className = "product-card";

    const nameDiv = document.createElement("div");
    nameDiv.className = "product-name";
    nameDiv.innerText = product.name;
    card.appendChild(nameDiv);

    const infoDiv = document.createElement("div");
    infoDiv.className = "product-info";

    const deptSpan = document.createElement("span");
    deptSpan.className = "department";
    deptSpan.innerText = `Department: ${product.department}`;
    infoDiv.appendChild(deptSpan);

    const aisleSpan = document.createElement("span");
    aisleSpan.className = "aisle";
    aisleSpan.innerText = `Aisle: ${product.aisle}`;
    infoDiv.appendChild(aisleSpan);

    card.appendChild(infoDiv);
    cardsContainer.appendChild(card);
  });

  msgDiv.appendChild(cardsContainer);
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

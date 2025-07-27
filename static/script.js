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

    // Process response with markdown
    appendMessage("bot", data.answer);
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

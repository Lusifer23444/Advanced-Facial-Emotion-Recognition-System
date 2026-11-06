# Advanced Facial Emotion Recognition System

An AI-powered real-time facial emotion recognition and emotional support web application built with **FastAPI**, **Google Gemini APIs**, and **Supabase**.

## 💡 The Story & Motivation

This project originated from my Diploma final-year project viva, where my team successfully trained a custom facial emotion recognition model using the FER 2013 dataset, achieving a solid 71% accuracy rate via Google Colab. During the viva defense, when questioned about accuracy limits, I realized that while custom models are great, relying purely on static models has its limits. That spark motivated me to build something much more advanced on my own.

As I started developing this as a side project over the last few months, it organically evolved into a full-stack development testbed. 

## 🚀 Key Challenges & Architecture Evolution

* **Solving User Retention:** Standard emotion detection suffers from poor user retention—people try it once and leave. To fix this, I integrated an **AI Emotional Support Chatbot** powered by Google Gemini to actively converse with users based on their detected emotional state.
* **Adding Real-World Utility:** Since the system captures live visual feeds, I expanded its utility by adding **Fashion and Grooming Recommendations**, transforming a basic academic prototype into a product-minded application.
* **Hybrid Architecture:** Instead of a heavy standalone local model, PyTorch, OpenCV, and NumPy act as a lightweight local bridge that processes and converts input data into clean JSON format. This JSON payload is then passed directly to the Gemini API, making the system highly resourceful.
* **Overcoming Roadblocks:** Navigating through dead zones—such as fixing local model response glitches and handling abrupt Google SDK shifts (transitioning to newer flash models and fixing SDK buffers)—tested my system architecture skills, ultimately leading to a robust Supabase database integration for secure emotion logging.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI & APIs:** Google Gemini API (`google-genai`)
* **Database & Logging:** Supabase PostgreSQL (`emotion_logs` table)
* **Frontend:** Single Page Application (HTML5, JavaScript)
* **Containerization & Deployment:** Docker, Railway

## ⚙️ Environment Variables (.env)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

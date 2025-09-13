# chatbot.py
"""
Enhanced Chatbot Module (PyQt6 Compatible)
Handles natural language processing and conversation management.
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Optional


class ChatbotManager:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.session_context = {}

        # Enhanced response patterns
        self.responses = {
            'greetings': [
                "Hello! It's great to see you! How can I help you today?",
                "Hi there! I'm excited to chat with you. What's on your mind?",
                "Greetings! I'm here and ready to assist you with anything you need.",
                "Hey! Welcome back! What would you like to talk about?",
            ],
            'personal_questions': [
                "That's a thoughtful question about me. I'm an AI assistant created to help and chat with people like you.",
                "I'm here to be helpful and engaging. What I find most interesting is learning about the people I talk with!",
                "As an AI, I don't have personal experiences like humans do, but I enjoy our conversations and helping however I can.",
            ],
            'compliments': [
                "Thank you so much! That means a lot to me. You seem pretty awesome yourself!",
                "That's very kind of you to say! I really appreciate it.",
                "You're too kind! I'm just happy I can be helpful to you.",
            ],
            'questions': [
                "That's a fascinating question! Let me think about that for a moment...",
                "Great question! I'd be happy to help you explore that topic.",
                "Interesting! That's something worth discussing in detail.",
                "I love thoughtful questions like that. Here's what I think...",
            ],
            'help_requests': [
                "I'm absolutely here to help! What specifically can I assist you with?",
                "Of course! I'd be delighted to help you with that.",
                "That's exactly what I'm here for! Let me see how I can help.",
            ],
            'goodbye': [
                "Goodbye! It was wonderful chatting with you. Have an amazing day!",
                "Take care! I really enjoyed our conversation. Until next time!",
                "See you later! Thanks for the great chat - come back anytime!",
                "Farewell! Hope to see you again soon!",
            ],
            'weather': [
                "I don't have access to current weather data, but I'd recommend checking a weather app or website for the most accurate information!",
                "For the most up-to-date weather information, I'd suggest checking your local weather service or a reliable weather app.",
            ],
            'time': [
                "I don't have access to the current time, but you can check your device's clock or ask your system for the time!",
                "For the current time, I'd recommend checking your computer or phone's clock display.",
            ],
            'appreciation': [
                "You're very welcome! I'm so glad I could help.",
                "It's my pleasure! That's what I'm here for.",
                "Happy to help! Feel free to ask if you need anything else.",
            ],
            'default': [
                "That's interesting! Tell me more about what you're thinking.",
                "I'd love to hear more about that. What's your perspective?",
                "That's a good point. What made you think of that?",
                "Hmm, that's worth exploring. What's your experience with that?",
                "I find that topic fascinating. What's your take on it?",
            ]
        }

        # Contextual keywords for better response matching
        self.keywords = {
            'greetings': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'personal_questions': ['who are you', 'what are you', 'tell me about yourself', 'your name'],
            'compliments': ['thank you', 'thanks', 'good job', 'well done', 'excellent', 'amazing', 'awesome'],
            'questions': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'help_requests': ['help', 'assist', 'support', 'can you', 'could you', 'would you'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'later', 'take care'],
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'cloudy', 'storm'],
            'time': ['time', 'clock', 'hour', 'minute', 'what time'],
            'appreciation': ['thank you', 'thanks', 'appreciate', 'grateful']
        }

    def get_response(self, message: str, user_name: Optional[str] = None) -> str:
        """Generate intelligent response to user message"""
        # Add to conversation history
        self.conversation_history.append({
            'user': user_name or 'Unknown',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Update session context
        if user_name:
            if user_name not in self.session_context:
                self.session_context[user_name] = {'message_count': 0, 'topics': []}
            self.session_context[user_name]['message_count'] += 1

        message_lower = message.lower().strip()

        # Handle empty messages
        if not message_lower:
            return "I didn't catch that. Could you please say something?"

        # Determine response category
        response_category = self._categorize_message(message_lower)

        # Get base response
        if response_category in self.responses:
            base_response = random.choice(self.responses[response_category])
        else:
            base_response = random.choice(self.responses['default'])

        # Personalize response if user is known
        if user_name and response_category != 'goodbye':
            # First time greeting
            if (response_category == 'greetings' and
                    self.session_context[user_name]['message_count'] == 1):
                base_response = f"Hello {user_name}! " + base_response

            # Add personal touch to other responses
            elif response_category in ['questions', 'help_requests']:
                base_response += f" What do you think, {user_name}?"

        return base_response

    def _categorize_message(self, message: str) -> str:
        """Categorize message based on keywords and patterns"""
        message_words = message.split()

        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message:
                    score += 1
            category_scores[category] = score

        # Find best matching category
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category

        # Special patterns
        if '?' in message:
            return 'questions'

        if any(word in message for word in ['help', 'assist', 'support']):
            return 'help_requests'

        return 'default'

    def get_conversation_history(self, user_name: Optional[str] = None) -> List[Dict]:
        """Get conversation history for a specific user or all users"""
        if user_name:
            return [msg for msg in self.conversation_history if msg['user'] == user_name]
        return self.conversation_history

    def get_user_stats(self, user_name: str) -> Dict:
        """Get statistics for a specific user"""
        if user_name not in self.session_context:
            return {'message_count': 0, 'topics': []}
        return self.session_context[user_name]

    def clear_history(self, user_name: Optional[str] = None):
        """Clear conversation history"""
        if user_name:
            self.conversation_history = [
                msg for msg in self.conversation_history if msg['user'] != user_name
            ]
            if user_name in self.session_context:
                del self.session_context[user_name]
        else:
            self.conversation_history = []
            self.session_context = {}

    def save_conversation(self, filename: Optional[str] = None):
        """Save conversation history to file"""
        if not filename:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w') as f:
                json.dump({
                    'conversation_history': self.conversation_history,
                    'session_context': self.session_context,
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False

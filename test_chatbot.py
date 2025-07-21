from app import KKHChatbot

# Test the new direct answer system
def test_chatbot():
    chatbot = KKHChatbot()
    
    # Test questions
    test_questions = [
        'What is the normal heart rate for a 2-year-old?',
        'What are the signs of dehydration?', 
        'How much fluid bolus for shock?',
        'What is the CPR ratio for children?',
        'Normal respiratory rate for infants?'
    ]
    
    print('Testing new direct answer system:')
    print('=' * 50)
    
    for question in test_questions:
        try:
            answer = chatbot.chat_with_lm_studio(question)
            print(f'Q: {question}')
            print(f'A: {answer}')
            print('-' * 30)
        except Exception as e:
            print(f'Error with question "{question}": {e}')
            print('-' * 30)

if __name__ == "__main__":
    test_chatbot()

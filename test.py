#Test file for Vertex AI Code Review Bot
print("Hello, world!")


# ❌ 1. Використання незахищеного вводу
user_input = input("Enter a number: ")
print("Result:", int(user_input) * 2)

# ❌ 2. Харкоджений пароль (погана практика)
password = "123456"
if password == "123456":
    print("Weak password in code!")

# ❌ 3. Функція без докстрінгу і без обробки винятків
def divide(a, b):
    return a / b

print(divide(10, 0))




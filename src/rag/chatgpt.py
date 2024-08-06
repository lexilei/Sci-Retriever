from openai import OpenAI
client = OpenAI()
file_name = "output.txt"

# Read the file content as a string
with open(file_name, 'r') as file:
    file_content = file.read()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "Answer the folowing question based on the information given."},
    {"role": "user", "content": f"In 3D point cloud classification for ScanObjectNN, what method does the DeLA paper introduce? {file_content}"}
  ]
)

print(completion.choices[0].message)
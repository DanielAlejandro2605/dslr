import json

def read_json(file_path):
	with open(file_path, 'r') as file:
		data = json.load(file)
	high_score = max(data, key=lambda x:x['score'])
	print(high_score)
	weight = high_score['weights']
	return weight

def main():
	file_path = 'evaluation_logs.json'
	weight = read_json(file_path)
	print(weight)

if __name__ == "__main__":
    main()

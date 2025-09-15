import numpy as np

rng = np.random.default_rng(seed=42)


#Simulate Dataset
num_entities = 100

feature1 = rng.integers(50, 101, size=num_entities)
feature2 = rng.integers(10, 51, size=num_entities)
feature3 = rng.integers(10, 501, size=num_entities)

data = np.column_stack((feature1, feature2, feature3))
print("Dataset shape: ", data.shape)
print("First 5 rows\n", data[:5])

#Basic Statics 

means = np.mean(data, axis=0)
medians = np.median(data, axis=0)
variances = np.var(data, axis=0)
std_devs = np.std(data, axis=0)

print(f"Means: {means}\nMedians: {medians}\nVariances: {variances}\nStandard Deviations: {std_devs}\n")

#Normalize Features

#Max-Min normalization

min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)

normalized_data1 = (data - min_vals) / (max_vals - min_vals)
print("Normaized_Data1 (first 5 rows):\n", normalized_data1[:5])

#Z normalization

normalized_data2 = (data - means) / variances
print("Normaized_Data1 (first 5 rows):\n", normalized_data2[:5])

#Weighted Composite Score

weights = np.array([0.5, 0.3, 0.2])
scores = np.dot(normalized_data1, weights)
print("First 5 composite scores:", scores[:5])

#Ranking Entities

rank_indices = np.argsort(scores)[::-1]
top_10 = scores[rank_indices[:10]]

print("Scores of top 10:", top_10)

#Boolean Indexing 

high_perf_engage = (normalized_data1[:,0] > 0.8) & (normalized_data1[:, 1] > 0.7)
selected_entities = np.where(high_perf_engage)[0]

print("Entities with high performance & engagement:", selected_entities)

#Advanced Matrix Operations

projection_matrix = np.array([[0.7, 0.2, 0.1],
                              [0.1, 0.6, 0.3]])
projected_scores = normalized_data1.dot(projection_matrix.T)

print("Projected scores shape:", projected_scores.shape)
print("First 5 projected scores:\n", projected_scores[:5])

#save

np.save("ai_data_intelligence_phase1.npy", data)
np.save("ai_data_normalized.npy", normalized_data1)
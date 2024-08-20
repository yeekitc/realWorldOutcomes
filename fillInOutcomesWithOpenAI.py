# Given a CSV file with a 2D matrix of objects and targets, where rows are objs 
# and cols are targets, and cells contain potential interactions between them,
# use OpenAI's GPT-3 to fill in the outcomes of the interactions in the empty row 
# below each outcome.

# TODO: Iterate upon prompt to get better outcomes
# TODO: csv, prompt, constraints

from openai import OpenAI
import pandas as pd
import time

####################
# Main
####################

# 1. Initialize OpenAI client
client = OpenAI()

# 2. Load DataFrame
csv_filename = "adjMatrix.csv"
df = pd.read_csv(csv_filename)
start_time = time.time()

# 3. Process DataFrame by filling in outcomes, sending each interaction to OpenAI
#    in a new API call, skipping the first two rows (target obj names and their states)
#    Every other row contains the object and its interactions with the targets
for index, row in df.iloc[2:].iterrows():
    if index % 2 == 1:
        # Check each cell in the row, skipping the first two columns (object names and their states)
        for col in df.columns[2:]:
            if pd.notna(row[col]):
                # Extract object, state, target, and target state
                interaction = row[col]
                obj = row[0]
                objState = row[1]
                target = col
                targetState = df.at[1, col]
                scenario = "You find yourself in a world where a dangerous dragon is terrorizing the kingdom."
                constraints = (
                    f"Constraints:"
                )
                
                prompt = (
                    f"You are an expert in predicting outcomes of interactions between various objects. "
                    f"Given the scenario: '{scenario} {constraints}', consider the following interaction: "
                    f"An object '{obj}', which is currently in the state '{objState}', interacts with a target '{target}', "
                    f"which is in the state '{targetState}', by performing the action '{interaction}'. "
                    f"What is the likely outcome of this interaction? Please provide the outcome in the form of: "
                    f"[1] A realistic outcome based on physical laws and [2] A narrative outcome based on the scenario context."
                )
                # Call the OpenAI API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are well-versed in the outcomes of interactions between objects."},
                        {"role": "user", "content": f"{scenario} What happens when you try to fix this problem by first having a {obj} that is {objState} interact with {target} that is {targetState} by the interaction '{interaction}'? \
                                                        Please provide the outcome concisely, in the form [1] <outcome> and [2] <outcome>, where [1] is the outcome constrained by reality and [2] is the outcome constrained by the scene. \
                        "},
                    ]
                )
                
                outcome = response.choices[0].message.content
                
                # Write the outcome below the interaction
                df.at[index + 1, col] = f"{outcome}"

# 4. Save DataFrame under a new filename: output_<original_filename>
output_filename = f"output_6_{csv_filename}"
df.to_csv(output_filename, index=False)
print(f"Processed DataFrame saved to {output_filename}")

end_time = time.time()
print(f"Total time taken: {end_time-start_time} seconds")

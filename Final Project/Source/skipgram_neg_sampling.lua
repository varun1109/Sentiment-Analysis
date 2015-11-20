require 'torch'
require 'nn'

-- Step 1: Define your vocabulary map
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

-- tests the functions above
local file = 'dict.txt'
local lines = lines_from(file)

dictionary = {}
-- print all line numbers and their contents
final_word = {}
count = 0
for k,v in pairs(lines) do
  --print (v)
  j=1
  count =count +1
  for word in string.gmatch(v, '([^ ]+)') do
    --print(word)
    	
    if j==2 then 
      dictionary[i] = word
      j=1
    else
      i = word
      j= j+1
    end    
    
  end 
--print (k)
end


print (count)
-- Step 2: Define constants
vocab_size=count
word_embed_size=5
learning_rate=0.01
window_size=3
max_epochs=5

neg_samples_per_pos_word=1
-- Step 3: Prepare your dataset
local file = 'sswe_token.txt'
local lines = lines_from(file)

data ={}
context_count =1

for k,v in pairs(lines) do
  --print (v)
  j=0
  for word in string.gmatch(v, '([^ ]+)') do
    j =j+1
    if j==3 then
      word1=torch.Tensor{dictionary[i1]}
      context1=torch.Tensor{dictionary[i],dictionary[word],dictionary[torch.random(count)]} -- P(i, nlp | like) (Note: 'dl' is a sample negative context for 'like')
      label1=torch.Tensor({1,1,0}) -- 0 denotes negative samples; 1 denotes the positve pairs		
      j=2
      i= i1
      i1 = word
      data[context_count]={{context1,word1},label1}
      context_count = context_count +1
    
      --print ('varun') 
       
    else if j==2 then
      i1 = word
    else
      i= word
    end
    end 	 	
    --print(word)
  end 
  --print (k)
end
function data:size() return context_count-1 end


-- Step 4: Define your model
wordLookup=nn.LookupTable(vocab_size,word_embed_size)
contextLookup=nn.LookupTable(vocab_size,word_embed_size)

model=nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(contextLookup)
model.modules[1]:add(wordLookup)
model:add(nn.MM(false,true)) -- 'true' to transpose the word embeddings before matrix multiplication
model:add(nn.Sigmoid())

-- Step 5: Define the loss function (Binary cross entropy error)
criterion=nn.BCECriterion()

-- Step 6: Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

print('Word Lookup before learning')
--print(wordLookup.weight)

-- Step 7: Train the model with dataset
trainer:train(data)

-- Step 8: Get the word embeddings
print('\nWord Lookup after learning')
print(wordLookup.weight)

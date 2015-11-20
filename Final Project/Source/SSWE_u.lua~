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
word_embed_size=50
learning_rate=0.01
window_size=3
max_epochs=5


-- Step 3: Prepare your dataset
local file = 'sswe_token.txt'
local lines = lines_from(file)
dataset={}
data ={}
context_count =1
context_count1 =1
for k,v in pairs(lines) do
  --print (v)
  j=0
  for word in string.gmatch(v, '([^ ]+)') do
    j =j+1
    if j==3 then
      input=torch.Tensor{dictionary[i],dictionary[i1],dictionary[word]} 
      output=torch.Tensor{1}
      input1=torch.Tensor{dictionary[i],dictionary[word]} -- P(like | i, nlp)
      output1=torch.Tensor{dictionary[i1]}	
      j=2
      i= i1
      i1 = word
      data[context_count] ={input,output}
      context_count = context_count +1
      dataset[context_count1] ={input1,output1}
      context_count1 = context_count1 +1	
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
function dataset:size() return context_count1-1 end
print (context_count)

-- Step 4: Define your model
lookup=nn.LookupTable(vocab_size,word_embed_size)
model=nn.Sequential()
model:add(lookup)
model:add(nn.Mean())
model:add(nn.Linear(word_embed_size,vocab_size))
model:add(nn.LogSoftMax())

lookupclone = lookup:clone("weight","bias","gradWeight","gradbias")
model1=nn.Sequential()
model1:add(lookupclone)
model1:add(nn.Mean())
model1:add(nn.Linear(word_embed_size,2))
model1:add(nn.LogSoftMax())

-- Step 5: Define the loss function (Negative log-likelihood)
criterion = nn.CrossEntropyCriterion()


-- Step 6: Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

trainer1=nn.StochasticGradient(model1,criterion)
trainer1.learningRate=learning_rate
trainer1.maxIteration=max_epochs

--print('Word Lookup before learning')
--print(model.modules[1].weight)

-- Step 7: Train the model with dataset
trainer:train(dataset)
trainer1:train(data)

-- Step 8: Get the word embeddings
--print('\nWord Lookup after learning')
print(model.modules[1].weight)
--print(model1.modules[1].weight)

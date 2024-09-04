# Test ollama and ollamar packages
# Ollama: https://ollama.com
# Ollamar: https://hauselin.github.io/ollama-r/

# Load libraries
library("tidyverse")
library("here")
source("R/setpaths.R")
# install.packages("ollamar")
library(ollamar) # require to start ollama server first on terminal

sharepoint_paths <- set_paths()

# Follow instructions from ollamar README


# Getting started ---------------------------------------------------------

test_connection()  # test connection to Ollama server
# if you see Ollama local server running, it's working

# generate a response/text based on a prompt; returns an httr2 response by default
resp <- generate("llama3.1", "tell me a 5-word story") 
resp

# <httr2_response>
#   POST http://127.0.0.1:11434/api/generate
# Status: 200 OK
# Content-Type: application/json
# Body: In memory (425 bytes)

# get just the text from the response object
resp_process(resp, "text") 
# get the text as a tibble dataframe
resp_process(resp, "df") 

# alternatively, specify the output type when calling the function initially
txt <- generate("llama3.1", "tell me a 5-word story", output = "text")

# list available models (models you've pulled/downloaded)
list_models()  

# Generate a completion ---------------------------------------------------
resp <- generate("llama3.1", "Tomorrow is a...")  # return httr2 response object by default
resp

resp_process(resp, "text")  # process the response to return text/vector output

generate("llama3.1", "Tomorrow is a...", output = "text")  # directly return text/vector output
generate("llama3.1", "Tomorrow is a...", stream = TRUE)  # return httr2 response object and stream output
generate("llama3.1", "Tomorrow is a...", output = "df", stream = TRUE)
generate("llama3.1", "Tomorrow is a...", output = "df")

# image prompt
# use a vision/multi-modal model
# ollamar::pull("llava-phi3") # install model locally
generate("llava-phi3", "What is in the image?", images = here("inputs/Ebola_virus.jpg"), output = 'text')
# NOTE: takes 2/3 minutes to run on my computer and uses all my CPU :(
#> [1] "\nThe image features a close-up of some parasites on the skin."
generate("llava-phi3", "What is in the image?", images = here("inputs/ebola_virus_2.png"), output = 'text')
#> [1] "The image features a red background with blue worm-like creatures on it."


# Chat --------------------------------------------------------------------

messages <- create_message("what is the capital of australia")  # default role is user
resp <- chat("llama3.1", messages)  # default returns httr2 response object
resp  # <httr2_response>
resp_process(resp, "text")  # process the response to return text/vector output

# specify output type when calling the function
chat("llama3.1", messages, output = "text")  # text vector
chat("llama3.1", messages, output = "df")  # data frame/tibble
chat("llama3.1", messages, output = "jsonlist")  # list
chat("llama3.1", messages, output = "raw")  # raw string
chat("llama3.1", messages, stream = TRUE)  # stream output and return httr2 response object

# create chat history
messages <- create_messages(
  create_message("end all your sentences with !!!", role = "system"),
  create_message("Hello!"),  # default role is user
  create_message("Hi, how can I help you?!!!", role = "assistant"),
  create_message("What is the capital of Australia?"),
  create_message("Canberra!!!", role = "assistant"),
  create_message("what is your name?")
)
cat(chat("llama3.1", messages, output = "text"))  # print the formatted output
#> I don't have a personal name, but I'm here to assist and communicate with you!!!

# image prompt
messages <- create_message("What is in the image?", images =  here("inputs/Ebola_virus.jpg"))
# use a vision/multi-modal model
chat("llava-phi3", messages, output = "text")
#> [1] "\nThe image features a close-up of a skin cell with tape coming out of it, resembling a worm."]

# Streaming responses -----------------------------------------------------

messages <- create_message("Tell me a 1-paragraph story.")

# use "llama3.1" model, provide list of messages, return text/vector output, and stream the output
chat("llama3.1", messages, output = "text", stream = TRUE)
#> [1] "As the last star faded from the night sky, Luna made her way to the old, creaky pier on the outskirts of town. 
#> She had always been drawn to this place, where the moonlight danced across the water and the world seemed to slow 
#> its frantic pace. With a small wooden boat bobbing gently behind her, she stood at the edge of the dock and let out 
#> a soft sigh. It was here, under the watchful gaze of the night sky, that she would often come to think about the 
#> future – not with anxiety or hope, but with a quiet acceptance. As the stars above twinkled like diamonds scattered 
#> across velvet, Luna pushed off into the water, leaving behind the familiar world and stepping softly into her own private universe."

messages <- create_message("Tell me a 1-paragraph story in French.")
chat(model = "llama3.1", messages = messages, output = "text", stream = TRUE)  # same as above
# Voici une histoire :
#   
#>   Il faisait un beau soleil à Paris lorsque Léon se décida à faire ce qu'il avait toujours voulu : 
#>   aller voir l'édit du roi au Louvre. Il marcha pendant une bonne heure, son cœur battant la chamade, 
#>   jusqu'à arriver enfin place de la Concorde. À mesure qu'il approchait de l'hôtel de ville, il pouvait 
#>   sentir la magie qui se déversait des murs du bâtiment. Léon s'approcha doucement de la porte d'entrée et,
#>   avec une petite prière silencieuse, il poussa la porte pour entrer dans l'hôtel. Et c'est là qu'il la vit : 
#>   une magnifique toile représentant le roi qui lui-même était représenté en majesté. Léon resta bouche bée, 
#>   tellement inspiré par cette œuvre d'art qu'il oublia presque d'être un peuple au pays.
# 
#> (Note : je m'excuse si cela n'est pas parfaitement écrit en français "natif", car j'ai essayé de créer une
#>  histoire qui soit compréhensible pour les lecteurs de niveau débutant.)


# Embeddings --------------------------------------------------------------

embed("llama3.1", "Hello, how are you?")

# don't normalize embeddings
embed("llama3.1", "Hello, how are you?", normalize = FALSE)

# get embeddings for similar prompts
e1 <- embed("llama3.1", "Hello, how are you?")
e2 <- embed("llama3.1", "Hi, how are you?")

# compute cosine similarity
sum(e1 * e2)  # not equals to 1
sum(e1 * e1)  # 1 (identical vectors/embeddings)
sum(e2 * e2)  # 1 (identical vectors/embeddings)

# non-normalized embeddings
e3 <- embed("llama3.1", "Hello, how are you?", normalize = FALSE)
e4 <- embed("llama3.1", "Hi, how are you?", normalize = FALSE)

sum(e3 * e4)  # not equals to 1
sum(e3 * e3)  # not equals to 1
sum(e4 * e4)  # equals to 1 (by chance?)


# Format messages ---------------------------------------------------------

# create a chat history with one message
messages <- create_message(content = "Hi! How are you? (1ST MESSAGE)", role = "assistant")
# or simply, messages <- create_message("Hi! How are you?", "assistant")
messages[[1]]  # get 1st message

# append (add to the end) a new message to the existing messages
messages <- append_message("I'm good. How are you? (2ND MESSAGE)", "user", messages)
messages[[1]]  # get 1st message
messages[[2]]  # get 2nd message (newly added message)

# prepend (add to the beginning) a new message to the existing messages
messages <- prepend_message("I'm good. How are you? (0TH MESSAGE)", "user", messages)
messages[[1]]  # get 0th message (newly added message)
messages[[2]]  # get 1st message
messages[[3]]  # get 2nd message

# insert a new message at a specific index/position (2nd position in the example below)
# by default, the message is inserted at the end of the existing messages (position -1 is the end/default)
messages <- insert_message("I'm good. How are you? (BETWEEN 0 and 1 MESSAGE)", "user", messages, 2)
messages[[1]]  # get 0th message
messages[[2]]  # get between 0 and 1 message (newly added message)
messages[[3]]  # get 1st message
messages[[4]]  # get 2nd message

# delete a message at a specific index/position (2nd position in the example below)
messages <- delete_message(messages, 2)

# create a chat history with multiple messages
messages <- create_messages(
  create_message("You're a knowledgeable tour guide.", role = "system"),
  create_message("What is the capital of Australia?")  # default role is user
)
chat("llama3.1", messages, output = "text")

#> [1] "That's an easy one to start with! The capital of Australia is Canberra, 
#> which is located in the Australian Capital Territory (ACT). However, it's worth 
#> noting that many people often associate Sydney as being the most famous city and 
#> the location where you'll find the iconic Opera House. But, when it comes to 
#> government and politics, Canberra is where the action happens.\n\nShall we 
#> move on to the next stop? I can tell you all about the fascinating history
#>  of Canberra or perhaps give you some tips on what to do and see in thi
#>  beautiful city!"


# create a list of messages 

# convert to dataframe
df <- dplyr::bind_rows(messages)  # with dplyr library
df <- data.table::rbindlist(messages)  # with data.table library

# convert dataframe to list with apply, purrr functions
apply(df, 1, as.list)  # convert each row to a list with base R apply
purrr::transpose(df)  # with purrr library


#%%
import torch
from datasets import load_dataset

# %%
# This is a huggingface dictionary like object see comments below for details: 
def get_dataset(dataset_name: str=None):
    """
    We will be working with dataset downloads from Huggingface. I thought it would be useful to get a basic understanding of the objects returned by Huggingface.

    load_dataset() returns a Huggingface DatasetDict object which is a dictionary like object. Here is a primer on Huggingface DatasetDict object:

        DatasetDict object:
        The DatasetDict in Hugging Face's datasets library is a container for multiple Dataset objects, typically representing different splits of a dataset like "train," "validation," and "test." It acts like a Python dictionary where keys are the names of the splits (e.g., "train", "validation", "test") and values are Dataset objects corresponding to those splits. 

        Key Differences vs Python dictionary:
            Purpose: A Python dictionary is a general data structure, while DatasetDict is specifically designed for managing and accessing different splits of a dataset within the Hugging Face ecosystem.

            Contents: A Python dictionary can store any key-value pairs. A DatasetDict specifically stores Dataset objects as its values, with keys typically representing the dataset split names (e.g., "train", "validation").

            Functionality: DatasetDict inherits methods from Dataset objects and provides functionalities tailored for data processing in machine learning, such as easy access to splits, mapping, filtering, and caching mechanisms for efficient handling of large datasets. Standard Python dictionaries do not inherently offer these specialized data processing functionalities.
            Memory Management: Dataset objects within a DatasetDict are often memory-mapped or loaded on demand, especially for large datasets, to efficiently manage memory. Python dictionaries typically load all data into memory directly.

            In summary:
            While a DatasetDict structurally resembles a Python dictionary (mapping keys to values), its specialized purpose and integration with the Hugging Face datasets library provide advanced features for managing and processing large, multi-split datasets for machine learning tasks. A standard Python dictionary is a more general-purpose data structure without these specialized functionalities.

    In our case, the DatasetDict object returned by raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k") has the following keys: 'train_sft', 'test_sft', 'train_gen', 'test_gen'. Each of these is a Dataset object. Note that we could have directly loaded one of the splits with load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"). Here is a primer in on the Dataset object:

        A Dataset object in Hugging Face is a highly optimized, in-memory data structure designed for efficient processing of large datasets, particularly in the context of machine learning. It can be thought of as a table-like structure, similar to a Pandas DataFrame, but with several key differences optimized for ML workloads. Each column has a specific data type, and each row represents one data point or example. .

        Here's a breakdown of its structure:
            Columns (Features): 
                The Dataset object contains various columns, each representing a "feature" of your data. These features can be of various types, including:
                    Primitive types: int, float, bool, string.
                    Nested structures: List, Array, ClassLabel, Sequence, Value.
                    Specialized types: Image, Audio, Video (for multimedia data).

                The feature schema defines the structure of each column. In ultrachat_200k, you'll have features with complex nested structures. For instance, a "messages" column might contain a list of dictionaries, where each dictionary has keys like "role" (whether it's a user or assistant message) and "content" (the actual text). The Dataset object understands and preserves these nested structures, so you don't lose information about the conversation flow.
            
            Rows (Examples): Each row in the Dataset represents a single data example. For the ultrachat_200k dataset, each row would represent one conversation or dialogue example

            Efficiency:
                One of the most important aspects of Dataset is its backend storage mechanism. It uses Apache Arrow, which is a columnar memory format. This means data is stored column-by-column rather than row-by-row. This columnar format is extremely efficient for the types of operations you typically do in ML - like filtering based on certain criteria, selecting specific columns, or applying transformations.
                The Dataset uses memory-mapping, which is crucial for handling large datasets. Instead of loading the entire dataset into RAM, it maps the file directly into your process's virtual memory space. The operating system then handles loading only the portions of data you actually access. This means you can work with datasets that are tens or hundreds of gigabytes even if you only have a few gigabytes of RAM available.
                The Dataset supports both in-memory and on-disk modes. For smaller datasets, it can operate entirely in memory for speed. For larger ones, it keeps data on disk and streams it as needed. You typically don't need to worry about this distinction as it handles it automatically.
                Each Dataset maintains metadata about itself - its cache files location, fingerprint (a hash representing its exact state and transformation history), and format. The fingerprint is particularly clever: it's used for caching transformed datasets. If you apply the same transformation twice, Hugging Face can recognize this and reuse the cached result rather than recomputing

            data access:
                The Dataset uses memory-mapping, which is crucial for handling large datasets. Instead of loading the entire dataset into RAM, it maps the file directly into your process's virtual memory space. The operating system then handles loading only the portions of data you actually access. This means you can work with datasets that are tens or hundreds of gigabytes even if you only have a few gigabytes of RAM available.
                When you access data from a Dataset, you can do so in multiple ways. You can access it like a dictionary using column names, you can access individual rows by index, you can slice it to get ranges of rows, and you can iterate through it in batches. The Dataset is lazy by default in many operations, meaning it doesn't actually compute or load data until you explicitly ask for it.

            methods for data manipulation and transformation include:
                map(): To apply a function to each example.
                filter(): To select specific examples based on a condition.
                shuffle(): To randomize the order of examples.
                select(): To select a subset of examples by index.
                shard(): To divide the dataset into smaller chunks.

                Datasets are immutable in a functional programming sense. When you apply transformations - like mapping a function over the dataset, filtering rows, or selecting columns - you don't modify the original Dataset. Instead, these operations return a new Dataset object. However, this doesn't mean the data is duplicated in memory. Hugging Face uses clever caching and referencing so that multiple Dataset objects can share the same underlying data storage efficiently.

            The Dataset object integrates seamlessly with PyTorch and TensorFlow. You can convert it to those frameworks' native formats, or you can use it directly with Hugging Face's Trainer API, which knows how to efficiently batch and feed data from a Dataset during training.
    """
    return load_dataset(dataset_name)

# help(get_dataset)
dataset_ultrachat = get_dataset("HuggingFaceH4/ultrachat_200k")

# %%
# examine type of objects returned by load_dataset
print(f'type of dataset_ultrachat: {type(dataset_ultrachat)}')
print(f'dataset_ultrachat keys: {dataset_ultrachat.keys()}')

# %%
# inspect the structure of the 'train_sft' split

print(f'type of dataset_ultrachat["train_sft"] : {type(dataset_ultrachat["train_sft"])} length: {len(dataset_ultrachat["train_sft"])}')

print(f'dataset_ultrachat["train_sft"] column_names: {dataset_ultrachat["train_sft"].column_names} ')

print(f'dataset_ultrachat["train_sft"] data features of each column: {dataset_ultrachat["train_sft"].features} ')

#%%
# experiment with various methods for accessing the data
"""
A primer on different methods for accessing data based on the example below.
When you use slice notation like [0:1], you're extracting data directly from the Dataset. The result is no longer a Dataset object - it's a dictionary where the keys are column names and the values are lists containing the actual data. So dataset_ultrachat["train_sft"][0:1] gives you a dictionary, and when you access ['messages'] from that dictionary, you get a list with one element (the messages from row 0).

When you use the select method with select([0]), you're creating a new Dataset object that contains only the rows you specified. It's still a full Dataset object with all the Dataset functionality, just with fewer rows. So dataset_ultrachat["train_sft"].select([0]) returns a Dataset with one row. When you then access ['messages'] on this Dataset, you're accessing the entire "messages" column, which returns a list containing the messages data from all rows in that Dataset object you created with .select(), in this case just the one row.

The end result - a list with the messages from row 0 - looks the same in both cases, but the path to get there is different.

The slice notation is more direct when you want to extract actual data values into Python native types. It's what you use when you're done with Dataset operations and want the raw data.

The select method is what you use when you want to continue working with Dataset operations - maybe you want to apply further transformations, filters, or pass it to other Hugging Face functions that expect a Dataset object.

And here is detail on other Dataset object methods:
The take method lets you grab the first N rows. It's similar to slicing from the beginning but more explicit in intent. It returns the actual data as a dictionary of lists, not a Dataset object.
The skip method does the opposite - it skips the first N rows and returns a new Dataset with the remaining rows. This is useful for splitting data or iterating through chunks.
The train_test_split method divides your Dataset into two new Dataset objects, typically for creating training and test sets. You can specify the split ratio or the exact size of the test set. This is handy when you need to create your own splits from a single Dataset.
The shard method divides the Dataset into multiple non-overlapping chunks (shards). You specify how many shards you want and which one to retrieve. This is particularly useful for distributed training where different processes need different portions of the data.
You can iterate directly over a Dataset using a for loop. Each iteration gives you one example as a dictionary. This is straightforward but loads data one row at a time.
The iter method with a batch size lets you iterate in batches. This is much more efficient than iterating row by row, especially during training. You get dictionaries where each value is a list containing the batch of data for that feature.
The to_pandas method converts the entire Dataset (or a slice of it) to a pandas DataFrame. This is useful if you want to leverage pandas' analysis and manipulation capabilities, though it does load data into memory.
The to_dict method converts the Dataset to a standard Python dictionary of lists. The to_list method converts it to a list of dictionaries (one dictionary per row). These are useful for getting native Python data structures.
You can also use the flatten method when you have nested structures. It unnests nested dictionaries or lists into separate columns with compound names, making the data more tabular.
The set_format method changes how data is returned when you access it. You can set it to return PyTorch tensors, TensorFlow tensors, NumPy arrays, or keep the default Python format. This is useful when integrating with training frameworks.
For column access specifically, you can use the column_names attribute to see what columns are available, and then access columns directly as if they were dictionary keys.
There's also with_format which temporarily changes the format for a specific operation without permanently modifying the Dataset.
The shuffle method randomly reorders the rows, which is essential before training to ensure your model doesn't learn from data order.
Each of these methods serves different use cases depending on whether you need the actual data values, want to continue with Dataset operations, need to integrate with other frameworks, or are preparing data for training.
"""

print(type(dataset_ultrachat["train_sft"][0:2]['messages']))
print(dataset_ultrachat["train_sft"][0:2]['messages'])
print(type(dataset_ultrachat["train_sft"].select(range(0,3))['messages']))
print(len(dataset_ultrachat["train_sft"].select(range(0,3))['messages']))

# the column_names attribute is very useful for determing the columns in each Dataset object

print(f"\nThe column names in each Dataset object:\n{dataset_ultrachat.column_names}\n")
print(f"The column names in the train_sft Dataset object:\n{dataset_ultrachat['train_sft'].column_names}")

# %%
import pprint

"""
each item (row) in 'train_sft' contains a multi-turn conversation. When you use slicing to extract a row, the object returned is a normal python dictionary with keys 'prompt', 'prompt_id', 'messages'. We are interested in the 'messages'. When you use slicing to access particular rows and the 'messages' column, a normal Python list of lists is returned where each nested list corresponds to a row.

When you use the .select() method to access rows, a Dataset object is returned. Then when you select the "messages" column from that object, a Huggingface "Column" object is returned. Here is primer on the Column object:
A Column object is Hugging Face's representation of a single column of data from a Dataset. It's essentially a wrapper around the underlying Arrow array that stores the actual column data.
When you access a column from a Dataset using dictionary-style notation (like ['messages']), you get back a Column object rather than a plain Python list. This Column object provides a view into the columnar data stored in Apache Arrow format.
The Column object is list-like in behavior - you can index into it, slice it, iterate over it, and check its length. When you perform these operations, it retrieves the actual data values from the Arrow backend. So if you iterate over a Column, you get the individual values from that column for each row.
One important characteristic is that a Column is still connected to the efficient Arrow storage. It doesn't immediately load all the data into Python objects. When you index into it or iterate, the conversion from Arrow format to Python objects happens on-demand.
You can think of the Column as an intermediary representation. It's not quite the raw Arrow data (which most users don't interact with directly), but it's also not yet converted to plain Python lists or values. It sits in between, providing a Pythonic interface while maintaining the efficiency benefits of the columnar storage.
When you perform operations on a Column - like converting it to a list, getting its length, or accessing specific elements - it knows how to efficiently retrieve that data from the Arrow arrays. For complex nested structures like the "messages" column in ultrachat (which contains lists of dictionaries), the Column preserves this structure.
If you want to convert a Column to a standard Python list, you can simply wrap it in the list() function. At that point, all the data is extracted from Arrow format and converted to native Python objects in memory.
The Column object is rarely something you need to think about explicitly. In most cases, you interact with it naturally as if it were a list, and it handles the underlying data access transparently. It's mainly an implementation detail that enables efficient data access while providing a familiar interface.

each message is a list of dictionaries. Each dictionary is a turn in a conversation. The keys of each dictionary are 'content' and 'role'.

"""

# use slicing to acces rows 0,1,2 of the train_sft split
example_slice = dataset_ultrachat['train_sft'][0:3]
print(f'type of example_slice: {type(example_slice)}. \nexample_slice keys: {example_slice.keys()}')

# Note that using slicing to access rows and then a column returns a normal Python list of lists. Each nested list corresponds to a row
example_slice_messages = dataset_ultrachat['train_sft'][0:3]['messages']
print(f'type of example_slice_messages: {type(example_slice_messages)} length: {len(example_slice_messages)}\n')


# use the .select() method to access row 0 of the train split
example_select = dataset_ultrachat['train_sft'].select(range(0,1))
print(f'type of example_select: {type(example_select)} \nexample_select features:{example_select.features}\nexample_select column_names:{example_select.column_names}\n')

# Note that using .select() method to access rows and then a column returns a Huggingface "Column" object which is a container that's similar to a Python list but a lot more efficient
example_select_messages = dataset_ultrachat['train_sft'].select(range(0,3))['messages']
print(f'type of example_select_messages: {type(example_select_messages)}\n')

# Here I just check that the first item in the list of lists created by slicing is the same as the first item in the Column container object created by using the .select() method
example_select_messages[0] == example_slice_messages[0]

#%%
# examine the messages. Each item in the column returned by example_select_messages is a converasttion with one or more turns. Each turn is a regular python dictionary with keys "role" and "content"
example_message = example_select_messages[0]
print(f'number of turns in example_message: {len(example_message)}\n')

# this is the first and second turn
print(f'example_msg {0} turn 0 type: {type(example_message[0])}   keys: {example_message[0].keys()}  length: {len(example_message[0])}\n {example_message[0]} \n')

print(f'example_msg {1} turn 1 type: {type(example_message[1])}   keys: {example_message[1].keys()}\n {example_message[0]} \n')

# %%
# take a look at the number of turns in the first 10 conversations
for example in dataset_ultrachat['train_sft'].select(range(0,10)):
    messages = example['messages']
    print(f'turns in this conversation: {len(messages)}')




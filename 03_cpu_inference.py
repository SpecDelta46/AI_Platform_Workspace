import os 
import time 
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM 
 
LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "models", "Qwen-0.5B-Chat") 
 
def run_cpu_inference(): 
    print("[*] Membangunkan Pustakawan AI (Loading Model ke RAM)...") 
    start_load = time.time() 
     
    # 1. Inisialisasi Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR) 
     
    # 2. Inisialisasi Model ke CPU 
    # Memuat file .safetensors ke memori RAM 
    model = AutoModelForCausalLM.from_pretrained( 
        LOCAL_MODEL_DIR, 
        device_map="cpu", # Memaksa penggunaan CPU 
        torch_dtype=torch.float32 # Presisi standar untuk CPU 
    ) 
     
    print(f"[+] Model berhasil dimuat dalam {time.time() - start_load:.2f} detik.\n") 
     
    # 3. Menyiapkan Prompt (Pertanyaan) 
    # Karena ini model chat, kita harus menggunakan format pesan 
 
    messages = [ 
        {"role": "system", "content": "Kamu adalah asisten HR yang ramah."}, 
        {"role": "user", "content": "Tolong buatkan satu kalimat motivasi untuk karyawan."} 
    ] 
     
    # Mengubah format pesan menjadi string yang dipahami Qwen, lalu di-tokenize 
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
add_generation_prompt=True) 
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to("cpu") 
     
    print("--- Pustakawan Sedang Mengetik ---") 
    start_infer = time.time() 
     
    # 4. Proses Inferensi (Generation) 
    # torch.no_grad() sangat penting untuk menghemat RAM saat inferensi 
    with torch.no_grad(): 
        generated_ids = model.generate( 
            model_inputs.input_ids, 
            max_new_tokens=50, # Maksimal kata yang dihasilkan 
            temperature=0.7,   # Tingkat kreativitas (0.1 kaku, 0.9 kreatif) 
            pad_token_id=tokenizer.eos_token_id 
        ) 
     
    # 5. Memisahkan pertanyaan dari jawaban, lalu dekode kembali ke teks 
    generated_ids = [ 
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, 
generated_ids) 
    ] 
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
     
    end_infer = time.time() 
    print(f"{response}\n") 
    print(f"[*] Waktu inferensi: {end_infer - start_infer:.2f} detik.") 
 
if __name__ == "__main__": 
    run_cpu_inference() 
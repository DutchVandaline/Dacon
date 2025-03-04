def deobfuscate_text(model, tokenizer, obf_text, desired_length=100):
    prompt = "Obfuscated: " + obf_text + " Normal:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    max_positions = model.config.n_positions
    available_tokens = max_positions - input_ids.shape[1]
    max_new = min(desired_length, available_tokens) if available_tokens > 0 else 0

    if max_new <= 0:
        print("Prompt length exceeds model limit.")
        return ""

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    new_token_ids = output_ids[0][input_ids.size(-1):]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return output_text.strip()

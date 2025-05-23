from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config


if __name__ == "__main__":

    config = Qwen2Config(
        hidden_size=896,
        num_attention_heads=14,
        num_hidden_layers=24,
        intermediate_size=4864,
        num_key_value_heads=2,
        max_window_layers=24,
    )
    model = Qwen2ForCausalLM(config)
    print(config)
    print(model)

    # Save the model
    model_path = "CustomQwen2Model"
    model.save_pretrained(model_path)
    config.save_pretrained(model_path)
    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer.bos_token_id = config.bos_token_id
    tokenizer.eos_token_id = config.eos_token_id
    tokenizer.pad_token_id = config.pad_token_id or tokenizer.pad_token_id
    tokenizer.save_pretrained(model_path)

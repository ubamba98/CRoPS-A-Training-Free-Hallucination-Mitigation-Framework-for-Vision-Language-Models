import math
def get_generations(self, 
                    input_ids, 
                    model_kwargs,
                    prepare_inputs_for_generation_fn, 
                    key_position,
                    output_attentions,
                    use_text_mask,
                    use_fast_v,
                    output_hidden_states):

    model_inputs = prepare_inputs_for_generation_fn(input_ids=input_ids, **model_kwargs)

    # prepare variable output controls (note: some models won't accept all output controls)
    model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
    model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})



    outputs = self(
        **model_inputs,
        return_dict=True,

        ## Additional arguments
        key_position=key_position,
        use_fast_v=use_fast_v,
        aggregate_layer_fast_v=model_kwargs.get("aggregate_layer_fast_v", None),
        minumum_fast_v_tokens=model_kwargs.get("minumum_fast_v_tokens", None),
        use_text_mask=use_text_mask,
        aggregate_layer_text_mask=model_kwargs.get("aggregate_layer_text_mask", None),
        minimum_text_tokens=model_kwargs.get("minimum_text_tokens", None),
    )

    # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=self.config.is_encoder_decoder,
    )
    return outputs, model_kwargs

def get_next_token_logits(outputs, input_ids):
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = outputs.logits[:, -1, :].clone().float()
    next_token_logits = next_token_logits.to(input_ids.device)
    return next_token_logits

def get_mask_text_rank(bias, time_step, alpha=2, lambda_=0.1):
    return math.floor(bias + alpha * (1 - math.exp(-lambda_ * time_step)))
## a reference to frequently used chat templates for convenience
class COMMON_CHAT_TEMPLATES:
    ### simple template which prepends a role header to the content
    simple_role_header = "{% for message in messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    ### passthrough template which just concatenates the content of the messages with no special tokens
    passthough_prompt_response = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

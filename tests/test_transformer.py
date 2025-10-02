# test_transformer.py

import pytest
import torch
import torch.nn.functional as F
from src.transformer import Transformer
from src.decoder import TransformerDecoder
from src.encoder import TransformerEncoder
import os
import platform


@pytest.mark.order(9)
def test_transformer_initialization():
    """Test if the Transformer model initializes correctly."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    assert model.encoder is not None
    assert model.decoder is not None
    assert model.output_linear is not None
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)


@pytest.mark.order(10)
def test_transformer_forward_pass_output_shape():
    """Test the forward pass of the Transformer model."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12

    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    attn_mask = None  # Assuming no mask for simplicity

    output = model(src_input, tgt_input, attn_mask)

    assert output.shape == (
        batch_size,
        tgt_seq_len,
        tgt_vocab_size,
    ), "Output shape mismatch"


@pytest.mark.order(11)
def test_transformer_generate_greedy_output_shape():
    """Test the generate method using greedy decoding."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="greedy"
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(12)
def test_transformer_generate_beam_search_output_shape():
    """Test the generate method using beam search decoding."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 1  # Beam search currently supports batch_size=1
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="beam_search", beam_size=3
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(13)
def test_transformer_generate_sampling_output_shape():
    """Test the generate method using multinomial sampling."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="sampling", temperature=1.0
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(14)
def test_transformer_generate_top_k_output_shape():
    """Test the generate method using top-k sampling."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="top_k", k=10
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(15)
def test_transformer_generate_top_p_output_shape():
    """Test the generate method using top-p sampling."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="top_p", p=0.9
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(16)
def test_transformer_generate_contrastive_output_shape():
    """Test the generate method using contrastive decoding."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 512
    num_heads = 8
    intermediate_size = 2048
    num_layers = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 1  # Contrastive decoding currently supports batch_size=1
    src_seq_len = 10
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="contrastive", k=5, alpha=0.6
    )

    assert (
        generated_sequence.shape[0] == batch_size
    ), "Batch size mismatch in generated sequences"
    assert (
        generated_sequence.shape[1] <= 15
    ), "Generated sequence length exceeds max_length"


@pytest.mark.order(17)
def test_transformer_generate_eos_termination():
    """Test that generation stops when EOS token is generated."""
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_position_embeddings = 50
    d_model = 4
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    # Define a simple Transformer model for the test
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers,
    )

    batch_size = 1
    src_seq_len = 5
    src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))

    # Mock the decoder to produce EOS token immediately
    def mock_decoder(tgt_input, enc_output):
        batch_size = tgt_input.size(0)
        seq_len = tgt_input.size(1)
        vocab_size = tgt_vocab_size
        # Generate logits that will result in EOS token
        logits = torch.full((batch_size, seq_len, vocab_size), float("-inf"))
        logits[:, -1, 3] = 0  # Assuming EOS_token = 3
        return logits

    # Replace the decoder's forward method with the mock
    model.decoder.forward = mock_decoder
    model.output_linear = torch.nn.Identity()

    generated_sequence = model.generate(
        src_input, max_length=15, decoding_strategy="greedy", EOS_token=3
    )

    assert generated_sequence.shape[1] == 1, "Generation did not stop at EOS token"


from unittest.mock import MagicMock
import torch.nn as nn


@pytest.fixture
def mock_transformer():
    # Create a Transformer instance
    model = Transformer(
        src_vocab_size=10,
        tgt_vocab_size=10,
        max_enc_position_embeddings=10,
        max_dec_position_embeddings=10,
        enc_d_model=8,
        dec_d_model=8,
        num_attention_heads=2,
        enc_intermediate_size=16,
        dec_intermediate_size=16,
        num_enc_hidden_layers=1,
        num_dec_hidden_layers=1,
    )

    # Mock encoder output
    encoder_output = torch.ones(
        (1, 5, 8)
    )  # Shape: (batch_size, src_seq_len, enc_d_model)

    # Create a mock encoder module
    class MockEncoder(nn.Module):
        def forward(self, *args, **kwargs):
            return encoder_output

    model.encoder = MockEncoder()

    # Mock decoder output
    def mock_decoder(tgt_input, enc_output):
        # Return a tensor of ones with appropriate shape
        seq_len = tgt_input.size(1)
        return torch.ones((1, seq_len, 8))  # Shape: (batch_size, seq_len, dec_d_model)

    class MockDecoder(nn.Module):
        def __init__(self, vocab_size=10, embed_dim=8):
            super(MockDecoder, self).__init__()
            # Define an embedding layer
            self.embeddings = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_dim
            )
            # Initialize embeddings to ones for simplicity
            self.embeddings.weight.data.fill_(1.0)

        def forward(self, tgt_input, enc_output):
            return mock_decoder(tgt_input, enc_output)

    model.decoder = MockDecoder()

    # Mock output linear layer to produce dynamic logits
    class MockOutputLinear(nn.Module):
        def __init__(self):
            super(MockOutputLinear, self).__init__()
            # Define different logits for each decoding step
            self.logits_list = [
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]]
                ),  # Step 1
                torch.tensor(
                    [[0.2, 0.3, 0.1, 0.4, 0.0, -0.2, -0.1, -0.3, -0.5, -0.4]]
                ),  # Step 2
                torch.tensor(
                    [[0.4, 0.1, 0.3, 0.2, 0.0, -0.3, -0.2, -0.1, -0.4, -0.5]]
                ),  # Step 3
                torch.tensor(
                    [[0.3, 0.2, 0.4, 0.1, 0.0, -0.4, -0.1, -0.3, -0.5, -0.2]]
                ),  # Step 4
                torch.tensor(
                    [[0.1, 0.4, 0.2, 0.3, 0.0, -0.5, -0.4, -0.3, -0.2, -0.1]]
                ),  # Step 5
            ]
            self.call_count = 0  # To track the number of calls (decoding steps)

        def forward(self, dec_output):
            # Ensure we stay within bounds of the logits list
            current_logits = self.logits_list[self.call_count]
            self.call_count = (self.call_count + 1) % len(
                self.logits_list
            )  # Increment the call count
            # Expand the logits to match the sequence length
            return current_logits.unsqueeze(1).expand(
                dec_output.size(0), dec_output.size(1), -1
            )  # Shape: (batch_size, seq_len, vocab_size)

    model.output_linear = MockOutputLinear()

    return model


def test_greedy_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])  # Example source input
    expected_tokens = [
        3,
        3,
        0,
        2,
        1,
    ]  # Based on mocked logits, argmax is at index 3 (value 0.4)

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="greedy",
        SOS_token=2,
        EOS_token=10,  # Assume EOS token is 0
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Greedy decoding did not produce expected tokens."


@pytest.mark.order(19)
def test_beam_search_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])
    expected_tokens = [3, 3, 1, 0, 3]

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="beam_search",
        beam_size=3,
        SOS_token=2,
        EOS_token=10,
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Beam search decoding did not produce expected tokens."


@pytest.mark.order(20)
def test_sampling_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])
    torch.manual_seed(0)  # Set seed for reproducibility

    # if "Microsoft" in platform.uname().release or platform.system() == "Windows":
    # # if os.name == 'nt':
    expected_tokens = [2, 8, 6, 3, 1]  # Based on sampling and the fixed logits
    # elif platform.system() == "Darwin" or platform.system() == "Linux":
    # # elif os.name == 'posix':
    #     expected_tokens = [6, 1, 2, 2, 1]

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="sampling",
        temperature=1.0,
        SOS_token=2,
        EOS_token=10,
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Sampling decoding did not produce expected tokens."


@pytest.mark.order(21)
def test_top_k_sampling_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])
    torch.manual_seed(0)

    # if "Microsoft" in platform.uname().release or platform.system() == "Windows":
    expected_tokens = [1, 0, 0, 2, 3]
    # elif platform.system() == "Darwin" or platform.system() == "Linux":
    #     expected_tokens = [1, 1, 0, 1, 1]

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="top_k",
        k=3,
        SOS_token=2,
        EOS_token=10,
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Top-k sampling decoding did not produce expected tokens."


@pytest.mark.order(22)
def test_top_p_sampling_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])
    torch.manual_seed(0)

    # if "Microsoft" in platform.uname().release or platform.system() == "Windows":
    expected_tokens = [1, 2, 6, 3, 3]
    # elif platform.system() == "Darwin" or platform.system() == "Linux":
    # expected_tokens = [6, 1, 3, 3, 3]

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="top_p",
        p=0.8,
        SOS_token=2,
        EOS_token=10,
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Top-p sampling decoding did not produce expected tokens."


@pytest.mark.order(23)
def test_contrastive_decoding(mock_transformer):
    src_input = torch.tensor([[1, 2, 3, 4, 5]])
    torch.manual_seed(0)

    expected_tokens = [3, 3, 0, 2, 1]

    generated_sequence = mock_transformer.generate(
        src_input,
        max_length=5,
        decoding_strategy="contrastive",
        k=5,
        alpha=0.6,
        SOS_token=2,
        EOS_token=10,
    )

    assert (
        generated_sequence.tolist()[0] == expected_tokens
    ), "Contrastive decoding did not produce expected tokens."

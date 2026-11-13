"""
Colormap lookup tables implemented with numpy only.

The two 256-entry uint8 RGB tables reproduce matplotlib's "turbo" (sequential)
and "RdBu_r" (diverging: blue = negative, white = zero, red = positive)
colormaps exactly as the previous matplotlib-based rendering produced them:
index = clip(floor(v * 256), 0, 255) and channel = trunc(c * 255).
Keeping them here removes matplotlib from the runtime image.
"""

from typing import Dict

import numpy as np

_TURBO_HEX = (
    "30123b31154232184a341b51351e5836215f37236538266c3929723a2c793b2f7f3c32853c358b3d37913e3a963f3d9c"
    "4040a14043a64145ab4148b0424bb5434eba4350be4353c24456c74458cb455bce455ed24560d64563d94666dd4668e0"
    "466be3466de64670e84673eb4675ed4678f0467af2467df4467ff64682f84584f94587fb4589fc448cfd438efd4291fe"
    "4193fe4096fe3f98fe3e9bfe3c9dfd3ba0fc39a2fc38a5fb36a8f934aaf833acf631aff52fb1f32db4f12bb6ef2ab9ed"
    "28bbeb26bde925c0e623c2e421c4e120c6df1ec9dc1dcbda1ccdd71bcfd41ad1d219d3cf18d5cc18d7ca17d9c717dac4"
    "17dcc217debf18e0bd18e1ba19e3b81ae4b61be5b41de7b11ee8af20e9ac22eba924eca627eda329eea02cef9d2ff09a"
    "32f19735f39438f4913bf48d3ff58a42f68746f7834af8804df97c51f97955fa7659fb725dfb6f61fc6c65fc6869fd65"
    "6dfd6271fd5f74fe5c78fe597cfe5680fe5384fe5087fe4d8bfe4b8efe4892fe4695fe4498fe429bfd409efd3ea1fc3d"
    "a4fc3ba6fb3aa9fb39acfa37aef937b1f836b3f835b6f735b9f534bbf434bef334c0f233c3f133c5ef33c8ee33caed33"
    "cdeb34cfea34d1e834d4e735d6e535d8e335dae236dde036dfde36e1dc37e3da37e5d838e7d738e8d538ead339ecd139"
    "edcf39efcd39f0cb3af2c83af3c63af4c43af6c23af7c039f8be39f9bc39f9ba38fab737fbb537fbb336fcb035fcae34"
    "fdab33fda932fda631fda330fea12ffe9e2efe9b2dfe982cfd952bfd9229fd8f28fd8c27fc8926fc8624fb8323fb8022"
    "fa7d20fa7a1ff9771ef8741cf7711bf76e1af66b18f56817f46516f36315f26014f15d13ef5a11ee5810ed550fec520e"
    "ea500de94d0de84b0ce6490be5460ae3440ae24209e04008de3e08dd3c07db3a07d93806d73606d63405d43205d23005"
    "d02f04ce2d04cb2b03c92903c72803c52602c32402c02302be2102bb1f01b91e01b61c01b41b01b11901ae1801ac1601"
    "a91501a61401a31201a011019d10019a0e01970d01940c01910b018e0a018b09018708018407018106027d05027a0402"
)

_DIVERGING_HEX = (
    "05306106326307346608366909386c0a3a6f0b3c720c3e750d40780e437b0f457e114781124984134b87144d8a154f8d"
    "1651901754931856951958981a5a9b1c5c9e1d5ea11e60a41f62a72064aa2166ac2368ad246aae256caf276db0286fb0"
    "2971b12b73b22c75b32d76b42f78b5307ab6317cb7337db8347fb93581b93783ba3884bb3986bc3b88bd3c8abe3d8bbf"
    "3f8dc0408fc14191c24393c34694c44996c54c98c64f9ac7529cc8559ec958a0ca5ba2cb5ea4cc61a6cd65a8ce68aacf"
    "6bacd06eaed171b0d274b2d377b4d57ab6d67db8d780bad884bcd987beda8ac0db8dc2dc90c4dd93c5de95c6df98c8df"
    "9ac9e09dcae19fcbe1a2cde2a4cee3a7cfe4a9d0e4abd2e5aed3e6b0d4e6b3d5e7b5d7e8b8d8e8bad9e9bddaeabfdceb"
    "c2ddebc4deecc7dfedc9e1edcce2eecee3efd1e5f0d2e5f0d3e6f0d5e7f0d6e7f1d8e8f1d9e9f1dbe9f1dceaf2deebf2"
    "dfecf2e1ecf3e2edf3e4eef3e5eef3e7eff4e8f0f4eaf1f4ebf1f4edf2f5eef3f5f0f3f5f1f4f6f3f5f6f4f5f6f6f6f6"
    "f7f6f6f7f5f4f7f4f2f7f3f0f8f2eef8f0ecf8efeaf8eee8f9ede7f9ece5f9ebe3f9eae1f9e9dffae8ddfae7dbfae5d9"
    "fae4d7fbe3d6fbe2d4fbe1d2fbe0d0fcdfcefcdeccfcddcafcdcc8fddbc7fcd8c4fcd6c1fbd4befbd2bcfbd0b9faceb6"
    "faccb4facab1f9c7aef9c5abf9c3a9f8c1a6f8bfa3f8bda1f7bb9ef7b99bf7b799f6b496f6b293f5b090f5ae8ef5ac8b"
    "f4aa88f4a886f4a683f3a380f2a07ef19e7cef9b7aee9878ed9676ec9374eb9072ea8e70e88b6ee7886ce6856ae58368"
    "e48065e27d63e17b61e0785fdf755dde725bdd7059db6d57da6a55d96853d86551d7624fd6604dd45d4bd35a4ad15749"
    "d05447ce5146cd4f45cc4c43ca4942c94641c7433fc6403ec53e3cc33b3bc2383ac03538bf3237be3036bc2d34bb2a33"
    "b92732b82431b6212fb51f2eb41c2db2192bb0172aad162aaa1529a71429a41328a112289e11279b1027991027960f26"
    "930e26900d258d0c258a0b24870a248409238108237e07227b06227805217504217203206f02206c011f69001f67001f"
)


TURBO_LUT: np.ndarray = np.frombuffer(bytes.fromhex(_TURBO_HEX), dtype=np.uint8).reshape(256, 3)
DIVERGING_LUT: np.ndarray = np.frombuffer(bytes.fromhex(_DIVERGING_HEX), dtype=np.uint8).reshape(256, 3)
TURBO_LUT.flags.writeable = False
DIVERGING_LUT.flags.writeable = False

COLORMAPS: Dict[str, np.ndarray] = {
    "turbo": TURBO_LUT,
    "diverging": DIVERGING_LUT,
    "rdbu_r": DIVERGING_LUT,
}


def get_lut(name: str) -> np.ndarray:
    """Return the LUT for a colormap name, falling back to turbo."""
    return COLORMAPS.get(str(name).lower(), TURBO_LUT)


def apply_colormap(values: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """
    Map an array of values in [0, 1] to uint8 RGB.

    Values outside [0, 1] are clipped; NaN/inf are treated as 0.

    Returns:
        Array of shape values.shape + (3,), dtype uint8.
    """
    lut = get_lut(colormap)
    v = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.clip(np.floor(v * 256.0), 0, 255).astype(np.intp)
    return lut[idx]

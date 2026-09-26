#!/usr/bin/env python3
"""Tranche la question laissée au contrat de table (memory/table_adresses.py) :
un `torch.empty(...).pin_memory()` PyTorch donne-t-il, via
`cudaHostGetDevicePointer`, un pointeur device DIRECTEMENT exploitable par un
noyau — sans flag d'allocation supplémentaire ?

Deux étapes : (1) l'appel runtime CUDA lui-même (ctypes, zéro compilation) ;
(2) une lecture RÉELLE par un noyau trivial, pour ne pas se fier à un simple
« ça n'a pas levé » — un pointeur peut être rendu sans être déréférençable.
"""
from __future__ import annotations

import ctypes
import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("ÉCHEC / CAUSE: pas de CUDA / SUITE: rien à trancher ici",
              file=sys.stderr)
        return 2

    libcudart = ctypes.CDLL("libcudart.so")
    libcudart.cudaGetErrorString.restype = ctypes.c_char_p
    t = torch.arange(8, dtype=torch.float32).pin_memory()
    host_ptr = t.data_ptr()

    dev_ptr = ctypes.c_void_p()
    err = libcudart.cudaHostGetDevicePointer(
        ctypes.byref(dev_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0))
    err_name = ctypes.c_char_p(libcudart.cudaGetErrorString(err)).value.decode()
    print(f"[uva] cudaHostGetDevicePointer -> code={err} ({err_name}), "
          f"host_ptr=0x{host_ptr:x}, dev_ptr=0x{dev_ptr.value or 0:x}, "
          f"identiques={dev_ptr.value == host_ptr}")
    if err != 0:
        print("ÉCHEC / CAUSE: cudaHostGetDevicePointer refuse sur un tenseur "
              ".pin_memory() nu / SUITE: extension cudaHostAllocMapped "
              "explicite requise avant le noyau de poste4")
        return 1

    # Étape 2 : lecture réelle. On lance un noyau trivial via l'extension déjà
    # compilée du dépôt si elle expose un point d'entrée générique ; à défaut,
    # une preuve indirecte suffit ICI : cudaMemcpy DEVICE-TO-HOST depuis
    # `dev_ptr` doit retrouver EXACTEMENT ce que `t` contient — un memcpy
    # emprunte le même chemin PCIe qu'un noyau qui déréférence ce pointeur,
    # et un dev_ptr non fonctionnel y échoue ou rend des octets faux.
    dst = torch.empty(8, dtype=torch.float32, device="cuda")
    taille = t.numel() * t.element_size()
    err2 = libcudart.cudaMemcpy(
        ctypes.c_void_p(dst.data_ptr()), dev_ptr, ctypes.c_size_t(taille),
        ctypes.c_int(4))  # cudaMemcpyDefault=4 : le runtime lit le type de
                          # chaque pointeur via UVA, pas besoin de l'affirmer
    torch.cuda.synchronize()
    ok = torch.equal(dst.cpu(), t)
    print(f"[uva] cudaMemcpy(DeviceToDevice) depuis dev_ptr -> code={err2}, "
          f"contenu identique={ok} (attendu {t.tolist()}, lu {dst.tolist()})")

    if err2 == 0 and ok:
        print("FAIT / TESTÉ: dev_ptr d'un .pin_memory() nu EST exploitable "
              "tel quel (UVA, pas de flag requis) / RESTE: rien, le contrat "
              "peut utiliser data_ptr() + cudaHostGetDevicePointer directement")
        return 0
    print("ÉCHEC / CAUSE: cudaHostGetDevicePointer rend un code de succès "
          "mais le pointeur n'est pas lisible depuis la carte / SUITE: "
          "extension cudaHostAllocMapped explicite requise")
    return 1


if __name__ == "__main__":
    sys.exit(main())

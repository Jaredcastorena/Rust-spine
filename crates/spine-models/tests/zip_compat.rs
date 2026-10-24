//! ZIP 9 compatibility coverage for the documented Candle backport.
use candle_core::{Device, Error, Tensor, npy::NpzTensors};

#[test]
fn zip9_npz_names_round_trip_utf8_and_cp437() {
    let directory = tempfile::tempdir().unwrap();
    for (name, cp437) in [("données", false), ("a", true)] {
        let file = directory.path().join("tensors.npz");
        let tensor = Tensor::new(&[2_u32, 4, 6], &Device::Cpu).unwrap();
        Tensor::write_npz(&[(name, &tensor)], &file).unwrap();
        let expected = if cp437 {
            let mut bytes = std::fs::read(&file).unwrap();
            let positions: Vec<_> = bytes
                .windows(5)
                .enumerate()
                .filter_map(|(i, value)| (value == b"a.npy").then_some(i))
                .collect();
            assert_eq!(positions.len(), 2);
            for position in positions {
                bytes[position] = 0x82;
            }
            std::fs::write(&file, bytes).unwrap();
            "é"
        } else {
            name
        };
        let eager = Tensor::read_npz(&file).unwrap();
        assert_eq!(eager.len(), 1);
        assert_eq!(eager[0].0, expected);
        assert_eq!(eager[0].1.to_vec1::<u32>().unwrap(), vec![2, 4, 6]);
        let lazy = NpzTensors::new(&file).unwrap();
        // The PTH loader traverses ZIP filenames too. An NPZ has no data.pkl,
        // so both borrowed UTF-8 and decoded CP437 names yield no PTH tensors.
        assert!(
            candle_core::pickle::read_pth_tensor_info(&file, false, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            lazy.get(expected)
                .unwrap()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap(),
            vec![2, 4, 6]
        );
    }
}

#[test]
fn malformed_npz_and_pth_return_errors() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("invalid.zip");
    std::fs::write(&file, b"not an archive").unwrap();
    assert!(matches!(Tensor::read_npz(&file), Err(Error::Zip(_))));
    assert!(matches!(NpzTensors::new(&file), Err(Error::Zip(_))));
    assert!(matches!(
        candle_core::pickle::read_pth_tensor_info(&file, false, None),
        Err(Error::Zip(_))
    ));
}

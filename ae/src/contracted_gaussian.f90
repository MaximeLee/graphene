module CotnractedGaussian
    type :: ContractedGaussian
        type(PrimitiveGaussian), dimension(:), allocatable:: PrimitiveGaussianArray
        double precision, dimension(:), allocatable:: coeffArray
        double precision, dimension(3) :: coeffArray
        contains
            procedure :: Initialize
            procedure :: Evaluate
            final :: FinalizeContractedGaussian
    end type ContractedGaussian
end module CotnractedGaussian

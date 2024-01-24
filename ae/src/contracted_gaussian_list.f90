module ContractedGaussianList
    use ContractedGaussian
    implicit none

    type :: ContractedGaussianList
        type(ContractedGaussian), dimension(:), allocatable:: ContractedGaussianArray
        double precision, dimension(:), allocatable:: coeffArray
        contains
            procedure :: Initialize
            procedure :: Evaluate
            final :: FinalizeContractedGaussianList
    end type ContractedGaussianList

contains
    subroutine Initialize(obj, CGArray, coeffArray)
        class(ContractedGaussianList), intent(inout) :: obj
        class(ContractedGaussian), dimension(:), intent(in) :: CGArray
        double precision, dimension(:), intent(in), optional :: coeffArray
        integer :: n

        n = size(CGArray)

        allocate(obj%contractedGaussianArray(n))
        allocate(obj%coeffArray(n))

        obj%ContractedGaussianArray = CGArray

        if present(coeffArray) then
            obj%coeffArray = coeffArray
        else then
            obj%coeffArray = 1.0
        endif
    end subroutine Initialize

    subroutine Evaluate(obj, X)
        class(ContractedGaussianList), intent(inout) :: obj
    end subroutine Evaluate

    subroutine FinalizeContractedGaussianList(obj)
        class(ContractedGaussianList), intent(inout) :: obj

        deallocate(obj%ContractedGaussianArray)
        deallocate(obj%coeffArray)
    end subroutine FinalizeContractedGaussianList

end module ContractedGaussianList

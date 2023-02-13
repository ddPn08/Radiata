import { ActionIcon, Flex, Input } from '@mantine/core'
import { IconArrowsLeftRight } from '@tabler/icons-react'
import { useAtom } from 'jotai'

import { generationParametersAtom } from '~/atoms/generationParameters'
import BetterNumInput from '~/components/ui/betterNumInput'
import { IMAGE_SIZE_STEP, MAX_IMAGE_SIZE, MIN_IMAGE_SIZE } from '~/utils/static'

const ImageSizeParameter = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)

  const swapWidthHeight = () => {
    setParameters((p) => ({
      ...p,
      image_width: p.image_height,
      image_height: p.image_width,
    }))
  }

  return (
    <Flex w={'100%'} align={'end'}>
      {/* Width */}
      <Input.Wrapper label={'Width'} w={'100%'}>
        <BetterNumInput
          defaultValue={parameters.image_width}
          value={parameters.image_width}
          min={MIN_IMAGE_SIZE}
          max={MAX_IMAGE_SIZE}
          step={IMAGE_SIZE_STEP}
          onChange={(e) => {
            if (e) {
              setParameters({ ...parameters, image_width: e })
            }
          }}
          allowWheel
        />
      </Input.Wrapper>

      <ActionIcon
        variant="outline"
        m={'sm'}
        color={'blue'}
        onClick={() => {
          swapWidthHeight()
        }}
      >
        <IconArrowsLeftRight size={16} />
      </ActionIcon>

      {/* Helight */}
      <Input.Wrapper label={'Height'} w={'100%'}>
        <BetterNumInput
          defaultValue={parameters.image_height}
          value={parameters.image_height}
          min={MIN_IMAGE_SIZE}
          max={MAX_IMAGE_SIZE}
          step={IMAGE_SIZE_STEP}
          onChange={(e) => {
            if (e) {
              setParameters({ ...parameters, image_height: e })
            }
          }}
          allowWheel
        />
      </Input.Wrapper>
    </Flex>
  )
}

export default ImageSizeParameter

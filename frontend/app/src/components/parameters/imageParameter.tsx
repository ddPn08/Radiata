import { Box, Button, CloseButton, FileButton, Group, Image, Stack, Text } from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { IconPhoto, IconUpload, IconX } from '@tabler/icons-react'
import { useAtom } from 'jotai'
import { useState } from 'react'

import NumberSliderInput from '../ui/numberSliderInput'

import { generationParametersAtom } from '~/atoms/generationParameters'
import { arrayBufferToBase64 } from '~/utils/base64'

const ImageParameter = () => {
  const [file, setFile] = useState<File | null>(null)
  const [parameters, setParameters] = useAtom(generationParametersAtom)

  const onFileChange = async (file: File | null) => {
    setFile(file)
    if (file) {
      const arrayBuffer = await file.arrayBuffer()
      const base64 = arrayBufferToBase64(arrayBuffer)
      setParameters((prev) => ({ ...prev, img: base64 }))
    } else {
      setParameters((prev) => ({ ...prev, img: '' }))
    }
  }

  return (
    <Stack>
      {file && (
        <NumberSliderInput
          label={'Strength'}
          defaultValue={parameters.strength}
          min={0.01}
          max={1.0}
          step={0.01}
          precision={2}
          onChange={(e) => {
            if (e) {
              setParameters((p) => ({ ...p, strength: e }))
            }
          }}
        />
      )}

      <FileButton onChange={onFileChange} accept={'image/*'}>
        {(props) => (
          <Button leftIcon={<IconUpload />} variant={'subtle'} {...props}>
            Upload image (or drag and drop)
          </Button>
        )}
      </FileButton>

      {file && (
        <Box pos={'relative'}>
          <Image src={URL.createObjectURL(file)} radius={'xs'} />
          <CloseButton
            variant={'filled'}
            color={'blue'}
            pos={'absolute'}
            m={'xs'}
            top={0}
            right={0}
            onClick={() => onFileChange(null)}
          />
        </Box>
      )}

      <Dropzone.FullScreen
        accept={IMAGE_MIME_TYPE}
        onDrop={(files) => {
          if (files.length > 0) {
            onFileChange(files[0] as File)
          }
        }}
        m={'xl'}
      >
        <Dropzone.Accept>
          <Group position={'center'} spacing={'xl'}>
            <IconUpload size={50} stroke={1.5} />

            <Box>
              <Text size="xl">Drop the image here!</Text>
              <Text size="sm"></Text>
            </Box>
          </Group>
        </Dropzone.Accept>

        <Dropzone.Reject>
          <Group position={'center'} spacing={'xl'}>
            <IconX size={50} stroke={1.5} />

            <Box>
              <Text size={'xl'}>This file type is not supported</Text>
              <Text size={'sm'}>(╯°□°)╯︵ ┻━┻</Text>
            </Box>
          </Group>
        </Dropzone.Reject>

        <Dropzone.Idle>
          <Group position={'center'} spacing={'xl'} h={'80%'}>
            <IconPhoto size={50} stroke={1.5} />

            <Box>
              <Text size="xl">Drag and drop your image here...</Text>
              <Text size="sm"></Text>
            </Box>
          </Group>
        </Dropzone.Idle>
      </Dropzone.FullScreen>
    </Stack>
  )
}

export default ImageParameter

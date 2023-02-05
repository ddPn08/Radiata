import { css, styled, useTheme } from 'decorock'
import type { BuildRequest } from 'internal:api'
import { createSignal, Show } from 'solid-js'
import { createStore } from 'solid-js/store'

import { api } from '~/api'
import { Button } from '~/components/ui/button'
import { CheckBox } from '~/components/ui/checkbox'
import { Input } from '~/components/ui/input'
import { Select } from '~/components/ui/select'
import { WithSlider } from '~/components/ui/slider'
import { HStack, VStack } from '~/components/ui/stack'

const Container = styled.div`
  display: flex;
  height: 100%;
  justify-content: center;
`

const Setting = styled.div`
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  margin-bottom: 1rem;
  gap: 0.5rem;

  & > div {
    width: 100%;
  }
`

export const Engine = () => {
  const theme = useTheme()
  const [req, setReq] = createStore({
    model_id: '',
    hf_token: '',
    fp16: false,
    opt_image_height: 512,
    opt_image_width: 512,
    max_batch_size: 1,
    onnx_opset: 16,
    build_static_batch: false,
    build_dynamic_shape: true,
    build_preview_features: false,
    force_engine_build: false,
    force_onnx_export: false,
    force_onnx_optimize: false,
    onnx_minimal_optimization: false,
  } as BuildRequest)
  const [status, setStatus] = createSignal<Record<string, any> | null>(null)

  return (
    <Container>
      <VStack
        class={css`
          width: 60%;
          ${theme.media.breakpoints.md} {
            width: 100%;
          }

          progress {
            width: 100%;
          }
        `}
      >
        <Show when={status()} keyed>
          {({ message, progress }) => (
            <div
              class={css`
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 0.25rem;
                border-radius: 0.5rem;
                background-color: ${theme.colors.secondary.darken(0.25)};
                font-family: 'Roboto Mono', 'Noto Sans JP', monospace;
                font-size: 0.9rem;
              `}
            >
              <div>
                {message} - {progress * 100}%
              </div>
              <div>This can take tens of minutes.</div>
            </div>
          )}
        </Show>
        <Button
          task={async () => {
            const { raw } = await api.buildEngineRaw({ buildRequest: { ...req } })
            const reader = raw.body?.getReader()
            if (!reader) return
            let done = true
            while (done) {
              const res = await reader.read()
              done = !res.done
              try {
                setStatus(JSON.parse(new TextDecoder().decode(res.value) || ''))
              } catch (_) {}
            }
            setStatus(null)
          }}
        >
          Build
        </Button>
        <Setting>
          <div>Model ID</div>
          <Input
            placeholder="huggingface model id (ex: CompVis/stable-diffusion-v1-4)"
            value={req.model_id || ''}
            onChange={(e) => setReq('model_id', e.currentTarget.value)}
          />
        </Setting>

        <Setting>
          <div>HuggingFace Access Token</div>
          <Input
            placeholder="hf_**********************************"
            value={req.hf_token || ''}
            onChange={(e) => setReq('hf_token', e.currentTarget.value)}
          />
        </Setting>

        <Setting>
          <WithSlider
            label="Optimization Image Height"
            max={1024}
            min={256}
            value={req.opt_image_height || 512}
            step={8}
            onChange={(v) => setReq('opt_image_height', parseInt(v))}
          />
        </Setting>

        <Setting>
          <WithSlider
            label="Optimization Image Width"
            max={1024}
            min={256}
            value={req.opt_image_width || 512}
            step={8}
            onChange={(v) => setReq('opt_image_width', parseInt(v))}
          />
        </Setting>

        <HStack>
          <Setting>
            <div>Denoising precision</div>
            <Select
              options={[
                { label: 'float32', value: 'fp32' },
                { label: 'float16', value: 'fp16' },
              ]}
              value={req.fp16 ? 'fp16' : 'fp32'}
              onChange={(v) => setReq('fp16', v.value === 'fp16')}
            />
          </Setting>
          <Setting>
            <div>Max batch size</div>
            <Input
              type="number"
              value={req.max_batch_size || 1}
              onInput={(e) => setReq('max_batch_size', parseInt(e.currentTarget.value))}
            />
          </Setting>
          <Setting>
            <div>Max batch size</div>
            <Input
              type="number"
              value={req.max_batch_size || 1}
              onInput={(e) => setReq('max_batch_size', parseInt(e.currentTarget.value))}
            />
          </Setting>
        </HStack>
        <HStack
          class={css`
            flex-wrap: wrap;
            gap: 1.5rem;
          `}
        >
          <CheckBox
            checked={req.build_static_batch || false}
            onChange={(e) => setReq('build_static_batch', e.currentTarget.checked)}
          >
            Build static batch
          </CheckBox>
          <CheckBox
            checked={req.build_dynamic_shape || false}
            onChange={(e) => setReq('build_dynamic_shape', e.currentTarget.checked)}
          >
            Build dynamic shape
          </CheckBox>
          <CheckBox
            checked={req.build_preview_features || false}
            onChange={(e) => setReq('build_preview_features', e.currentTarget.checked)}
          >
            Build preview features
          </CheckBox>
          <CheckBox
            checked={req.force_engine_build || false}
            onChange={(e) => setReq('force_engine_build', e.currentTarget.checked)}
          >
            Force engine build
          </CheckBox>
          <CheckBox
            checked={req.force_onnx_export || false}
            onChange={(e) => setReq('force_onnx_export', e.currentTarget.checked)}
          >
            Force onnx export
          </CheckBox>
          <CheckBox
            checked={req.force_onnx_optimize || false}
            onChange={(e) => setReq('force_onnx_optimize', e.currentTarget.checked)}
          >
            Force onnx optimize
          </CheckBox>
          <CheckBox
            checked={req.onnx_minimal_optimization || false}
            onChange={(e) => setReq('onnx_minimal_optimization', e.currentTarget.checked)}
          >
            Onnx minimal optimization
          </CheckBox>
        </HStack>
      </VStack>
    </Container>
  )
}
